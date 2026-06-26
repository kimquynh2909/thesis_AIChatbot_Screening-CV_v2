from __future__ import annotations

import os
from textwrap import dedent
from typing import Any

from src.llm.gemini_utils import DEFAULT_GEMINI_CHAT_MODEL, get_google_api_key, resolve_gemini_chat_model
from src.models.hybrid_matcher import recommendation_label
from src.utils.file_utils import shorten_text


EXPLANATION_TEMPLATE = """You are an HR screening assistant.
Use only the job description and resume evidence below.
Do not infer protected attributes, personality, culture fit, health, age, gender, race, ethnicity, religion, disability, or family status.
The ranking model already produced the score. Your task is only to explain it.

Job description:
{job_description}

Candidate resume:
{resume_text}

Model evidence:
- Final score: {final_score}%
- Semantic score: {semantic_score}%
- Recommendation label: {recommendation}
- Matched skills: {matched_skills}
- Missing required skills: {missing_skills}
- Experience detected in resume: {resume_years} years
- Experience required by JD: {required_years} years
- Retrieved Qdrant job metadata: {rag_evidence}

Write a concise HR-friendly explanation with these sections:
1. Overall assessment
2. Evidence supporting the match
3. Missing or weak evidence
4. Suggested HR follow-up questions
"""


def generate_candidate_explanation(result: Any, jd_text: str, resume_text: str, use_llm: bool = True) -> str:
    if use_llm:
        llm_output = _try_generate_with_gemini(result, jd_text, resume_text)
        if llm_output:
            return llm_output
    return deterministic_explanation(result)


def deterministic_explanation(result: Any) -> str:
    matched = ", ".join(result.matched_skills[:12]) or "No required dictionary skills were detected in both texts."
    missing = ", ".join(result.missing_skills[:12]) or "No required dictionary skills were missing from the dictionary analysis."
    label = recommendation_label(result.final_score)
    explanation = dedent(
        f"""
        Overall assessment: {label}. The candidate received a final match score of {result.final_score:.2f}% and a semantic similarity score of {result.semantic_score:.2f}%.

        Evidence supporting the match: Matched skills include {matched}. The resume shows approximately {result.detected_resume_years:g} years of experience, compared with {result.required_years:g} years required or detected from the JD.

        Missing or weak evidence: Missing required skills include {missing}. This does not prove the candidate lacks these skills; it only means the current resume text did not provide clear evidence.

        Suggested HR follow-up questions: Ask the candidate to clarify the missing skills, provide examples of relevant project impact, and confirm experience level or certifications where the resume is ambiguous.
        """
    ).strip()
    rag_evidence = _format_rag_evidence(result)
    if rag_evidence:
        explanation = f"{explanation}\n\nRetrieved Qdrant job metadata: {rag_evidence}"
    return explanation


def _try_generate_with_gemini(result: Any, jd_text: str, resume_text: str) -> str | None:
    api_key = get_google_api_key()
    if not api_key:
        return None

    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None

    prompt = PromptTemplate.from_template(EXPLANATION_TEMPLATE)
    model_name = resolve_gemini_chat_model(api_key, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_CHAT_MODEL))
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "job_description": shorten_text(jd_text, 5000),
                "resume_text": shorten_text(resume_text, 5000),
                "final_score": f"{result.final_score:.2f}",
                "semantic_score": f"{result.semantic_score:.2f}",
                "recommendation": result.recommendation,
                "matched_skills": ", ".join(result.matched_skills) or "None detected",
                "missing_skills": ", ".join(result.missing_skills) or "None detected",
                "resume_years": f"{result.detected_resume_years:g}",
                "required_years": f"{result.required_years:g}",
                "rag_evidence": _format_rag_evidence(result) or "None available",
            }
        )
        return getattr(response, "content", str(response)).strip()
    except Exception:
        return None


def _format_rag_evidence(result: Any, max_chunks: int = 3, max_chars: int = 280) -> str:
    evidence = getattr(result, "rag_evidence", None) or []
    lines = []
    for item in evidence[:max_chunks]:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        score = float(item.get("score", 0.0) or 0.0)
        job_id = item.get("job_id") or "unknown job"
        job_title = item.get("job_title") or "unknown title"
        job_category = item.get("category") or item.get("job_category") or "unknown category"
        required_skills = shorten_text(str(item.get("required_skills") or ""), max_chars)
        lines.append(
            f"{job_title} ({job_category}), job {job_id}, rank {rank}, "
            f"score {score:.4f}, required skills: {required_skills}"
        )
    return " | ".join(lines)
