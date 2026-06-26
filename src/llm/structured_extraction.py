from __future__ import annotations

import json
import os
import re
from typing import Any

from src.llm.gemini_utils import DEFAULT_GEMINI_CHAT_MODEL, get_google_api_key, resolve_gemini_chat_model
from src.models.hybrid_matcher import extract_required_years, extract_resume_years
from src.preprocessing.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import clean_for_matching
from src.utils.constants import DEGREE_KEYWORDS
from src.utils.file_utils import shorten_text


JD_EXTRACTION_KEYS = ["required_skills", "preferred_skills", "tools", "education", "experience"]
CV_EXTRACTION_KEYS = ["skills", "education", "experience", "projects"]

JD_EXTRACTION_PROMPT = """You are an information extraction engine for an HR screening system.
Use only the job description text. Do not infer protected attributes.
Return JSON only. Do not include Markdown, prose, or code fences.

Return exactly this JSON object shape:
{
  "required_skills": ["explicitly required technical or domain skills"],
  "preferred_skills": ["nice-to-have or preferred skills"],
  "tools": ["software tools, platforms, frameworks, libraries, cloud services"],
  "education": ["degree, major, certification, or education requirements"],
  "experience": ["years, seniority, role background, or responsibility requirements"]
}

Job description:
<<JOB_DESCRIPTION>>
"""

CV_EXTRACTION_PROMPT = """You are an information extraction engine for an HR screening system.
Use only the CV text and the supplied job description extraction. Do not infer protected attributes.
Return JSON only. Do not include Markdown, prose, or code fences.

Return exactly this JSON object shape:
{
  "skills": ["skills explicitly evidenced by the CV"],
  "education": ["degrees, majors, certifications, or education evidence in the CV"],
  "experience": ["role, years, domain, and responsibility evidence in the CV"],
  "projects": ["project names or project evidence in the CV"]
}

Job description extraction JSON:
<<JOB_EXTRACTION>>

CV text:
<<RESUME_TEXT>>
"""


def extract_job_description_json(
    jd_text: str,
    use_llm: bool = True,
    skill_extractor: SkillExtractor | None = None,
) -> dict[str, list[str]]:
    if use_llm:
        prompt = JD_EXTRACTION_PROMPT.replace("<<JOB_DESCRIPTION>>", shorten_text(jd_text, 7000))
        llm_output = _try_llm_json(prompt)
        if llm_output:
            return normalize_jd_extraction(llm_output)
    return deterministic_jd_extraction(jd_text, skill_extractor=skill_extractor)


def extract_cv_json(
    resume_text: str,
    jd_extraction: dict[str, Any] | None = None,
    use_llm: bool = True,
    skill_extractor: SkillExtractor | None = None,
) -> dict[str, Any]:
    normalized_jd = normalize_jd_extraction(jd_extraction or {})
    if use_llm:
        prompt = CV_EXTRACTION_PROMPT.replace("<<JOB_EXTRACTION>>", json.dumps(normalized_jd, ensure_ascii=False))
        prompt = prompt.replace("<<RESUME_TEXT>>", shorten_text(resume_text, 7000))
        llm_output = _try_llm_json(prompt)
        if llm_output:
            return normalize_cv_extraction(llm_output, normalized_jd)
    return deterministic_cv_extraction(resume_text, normalized_jd, skill_extractor=skill_extractor)


def extract_screening_json(
    jd_text: str,
    resumes: list[Any],
    use_llm: bool = True,
    skill_extractor: SkillExtractor | None = None,
) -> dict[str, Any]:
    extractor = skill_extractor or SkillExtractor()
    jd_extraction = extract_job_description_json(jd_text, use_llm=use_llm, skill_extractor=extractor)
    resume_extractions = []
    for index, resume in enumerate(resumes, start=1):
        text = str(getattr(resume, "text", ""))
        candidate_id = str(getattr(resume, "candidate_id", f"candidate_{index}"))
        filename = str(getattr(resume, "filename", candidate_id))
        resume_extractions.append(
            {
                "candidate_id": candidate_id,
                "filename": filename,
                "extraction": extract_cv_json(text, jd_extraction, use_llm=use_llm, skill_extractor=extractor),
            }
        )
    return {
        "job_description": jd_extraction,
        "resumes": resume_extractions,
    }


def normalize_jd_extraction(data: dict[str, Any]) -> dict[str, list[str]]:
    return {key: _unique_strings(data.get(key, [])) for key in JD_EXTRACTION_KEYS}


def normalize_cv_extraction(data: dict[str, Any], jd_extraction: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "skills": _unique_strings(data.get("skills", [])),
        "education": _unique_strings(data.get("education", [])),
        "experience": _unique_strings(data.get("experience", [])),
        "projects": _unique_strings(data.get("projects", [])),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def deterministic_jd_extraction(
    jd_text: str,
    skill_extractor: SkillExtractor | None = None,
) -> dict[str, list[str]]:
    extractor = skill_extractor or SkillExtractor()
    skills = extractor.extract(jd_text)
    preferred_terms = _extract_preferred_terms(jd_text, extractor)
    education = _education_signals(jd_text)
    experience = _experience_signals(jd_text, extract_required_years(jd_text))
    return {
        "required_skills": [skill for skill in skills if skill not in preferred_terms],
        "preferred_skills": preferred_terms,
        "tools": _tool_like_terms(skills),
        "education": education,
        "experience": experience,
    }


def deterministic_cv_extraction(
    resume_text: str,
    jd_extraction: dict[str, Any],
    skill_extractor: SkillExtractor | None = None,
) -> dict[str, Any]:
    extractor = skill_extractor or SkillExtractor()
    skills = extractor.extract(resume_text)
    education = _education_signals(resume_text)
    experience = _experience_signals(resume_text, extract_resume_years(resume_text))
    projects = _project_signals(resume_text)
    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "projects": projects,
    }


def _try_llm_json(prompt: str) -> dict[str, Any] | None:
    api_key = get_google_api_key()
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None

    model_name = resolve_gemini_chat_model(api_key, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_CHAT_MODEL))
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        return parse_json_object(content)
    except Exception:
        return None


def _unique_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("name") or item.get("requirement") or item.get("text") or item.get("value")
                if text:
                    items.append(str(text))
            else:
                items.append(str(item))
    else:
        items = [str(value)]

    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip(" -:\t\r\n")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _extract_preferred_terms(text: str, extractor: SkillExtractor) -> list[str]:
    preferred_lines = []
    for line in re.split(r"[\n.;]", text):
        lowered = line.lower()
        if any(marker in lowered for marker in ["preferred", "nice to have", "plus", "bonus", "advantage"]):
            preferred_lines.append(line)
    return extractor.extract(" ".join(preferred_lines)) if preferred_lines else []


def _tool_like_terms(skills: list[str]) -> list[str]:
    tool_markers = {
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",
        "spark",
        "hadoop",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pytorch",
        "keras",
        "langchain",
        "hugging face",
        "faiss",
        "chromadb",
        "qdrant",
        "power bi",
        "tableau",
        "excel",
    }
    return [skill for skill in skills if skill in tool_markers]


def _education_signals(text: str) -> list[str]:
    cleaned = clean_for_matching(text)
    return [keyword for keyword in DEGREE_KEYWORDS if keyword in cleaned]


def _experience_signals(text: str, years: float) -> list[str]:
    signals: list[str] = []
    if years > 0:
        signals.append(f"{years:g} years")
    for line in re.split(r"[\n.;]", text):
        lowered = line.lower()
        if "experience" in lowered or re.search(r"\b\d+(?:\.\d+)?\+?\s*(?:years|yrs)\b", lowered):
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned:
                year_match = re.search(r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", cleaned, flags=re.IGNORECASE)
                if year_match:
                    signals.append(f"{float(year_match.group(1)):g} years")
                signals.append(cleaned)
    return _unique_strings(signals)


def _project_signals(text: str) -> list[str]:
    signals = []
    for line in re.split(r"[\n.;]", text):
        lowered = line.lower()
        if "project" in lowered or "built" in lowered or "developed" in lowered or "implemented" in lowered:
            cleaned = re.sub(r"\s+", " ", line).strip()
            if cleaned:
                signals.append(cleaned)
    return _unique_strings(signals)

