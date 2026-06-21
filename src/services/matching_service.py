from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any

from src.llm.structured_extraction import extract_screening_json
from src.models.hybrid_matcher import HybridMatcher, HybridScoreBreakdown, recommendation_label
from src.preprocessing.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import clean_for_display, clean_for_matching
from src.utils.constants import MODEL_BGE, MODEL_BM25, MODEL_E5, MODEL_GLOVE, MODEL_GOOGLE, MODEL_SBERT, MODEL_TFIDF, MODEL_WORD2VEC


@dataclass
class ResumeDocument:
    candidate_id: str
    filename: str
    text: str


@dataclass
class MatchResult:
    rank: int
    candidate_id: str
    filename: str
    model_key: str
    model_name: str
    semantic_score: float
    final_score: float
    recommendation: str
    matched_skills: list[str]
    missing_skills: list[str]
    resume_skills: list[str]
    jd_skills: list[str]
    detected_resume_years: float
    required_years: float
    education_signals: list[str]
    certification_signals: list[str]
    runtime_seconds: float
    cleaned_resume_text: str
    breakdown: HybridScoreBreakdown
    structured_extraction: dict[str, Any] | None = None

    def to_export_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("cleaned_resume_text", None)
        data.pop("breakdown", None)
        data.pop("structured_extraction", None)
        for key in ["matched_skills", "missing_skills", "resume_skills", "jd_skills", "education_signals", "certification_signals"]:
            data[key] = ", ".join(data[key])
        return data


def get_matcher(model_key: str):
    model_key = model_key.lower()
    if model_key == MODEL_TFIDF:
        from src.models.tfidf_matcher import TfidfMatcher

        return TfidfMatcher()
    if model_key == MODEL_BM25:
        from src.models.bm25_matcher import BM25Matcher

        return BM25Matcher()
    if model_key == MODEL_WORD2VEC:
        from src.models.word2vec_matcher import Word2VecMatcher

        return Word2VecMatcher()
    if model_key == MODEL_GLOVE:
        from src.models.glove_matcher import GloveMatcher

        return GloveMatcher()
    if model_key == MODEL_SBERT:
        from src.models.sbert_matcher import SBERTMatcher

        return SBERTMatcher()
    if model_key == MODEL_E5:
        from src.models.e5_matcher import E5Matcher

        return E5Matcher()
    if model_key == MODEL_BGE:
        from src.models.bge_matcher import BGEMatcher

        return BGEMatcher()
    if model_key == MODEL_GOOGLE:
        from src.models.google_embedding_matcher import GoogleEmbeddingMatcher

        return GoogleEmbeddingMatcher()
    raise ValueError(f"Unknown matching model: {model_key}")


def match_resumes(
    jd_text: str,
    resumes: list[ResumeDocument],
    model_key: str,
    weights: dict[str, float] | None = None,
    use_hybrid_score: bool = True,
    skill_extractor: SkillExtractor | None = None,
    use_structured_extraction_score: bool = True,
    use_llm_extraction: bool = True,
    structured_extraction: dict[str, Any] | None = None,
) -> list[MatchResult]:
    if not jd_text or not jd_text.strip():
        raise ValueError("A job description is required before matching resumes.")
    if not resumes:
        raise ValueError("At least one resume is required before matching.")

    matcher = get_matcher(model_key)
    hybrid = HybridMatcher(skill_extractor=skill_extractor, weights=weights)
    cleaned_jd = clean_for_matching(jd_text)
    cleaned_resumes = [clean_for_matching(resume.text) for resume in resumes]

    started = time.perf_counter()
    semantic_scores = matcher.score(cleaned_jd, cleaned_resumes)
    total_runtime = time.perf_counter() - started
    per_resume_runtime = total_runtime / max(len(resumes), 1)
    extraction_payload = _resolve_extraction_payload(
        jd_text=jd_text,
        resumes=resumes,
        use_hybrid_score=use_hybrid_score,
        use_structured_extraction_score=use_structured_extraction_score,
        use_llm_extraction=use_llm_extraction,
        structured_extraction=structured_extraction,
        skill_extractor=skill_extractor,
    )
    jd_extraction = extraction_payload.get("job_description", {}) if extraction_payload else {}
    resume_extractions = _resume_extractions_by_identity(extraction_payload)

    results: list[MatchResult] = []
    for resume, resume_text, semantic_score in zip(resumes, cleaned_resumes, semantic_scores, strict=False):
        resume_extraction = resume_extractions.get(resume.candidate_id) or resume_extractions.get(resume.filename)
        if use_hybrid_score and use_structured_extraction_score and resume_extraction:
            breakdown = hybrid.explainable_score_from_extraction(semantic_score, jd_extraction, resume_extraction)
        else:
            breakdown = hybrid.explainable_score(semantic_score, resume_text, cleaned_jd)
        final_score = breakdown.final_score if use_hybrid_score else round(float(semantic_score) * 100.0, 2)
        results.append(
            MatchResult(
                rank=0,
                candidate_id=resume.candidate_id,
                filename=resume.filename,
                model_key=model_key,
                model_name=getattr(matcher, "name", model_key),
                semantic_score=round(float(semantic_score) * 100.0, 2),
                final_score=final_score,
                recommendation=recommendation_label(final_score),
                matched_skills=breakdown.matched_skills,
                missing_skills=breakdown.missing_skills,
                resume_skills=breakdown.resume_skills,
                jd_skills=breakdown.jd_skills,
                detected_resume_years=breakdown.detected_resume_years,
                required_years=breakdown.required_years,
                education_signals=breakdown.education_signals,
                certification_signals=breakdown.certification_signals,
                runtime_seconds=round(per_resume_runtime, 4),
                cleaned_resume_text=clean_for_display(resume.text),
                breakdown=breakdown,
                structured_extraction=resume_extraction,
            )
        )

    ranked = sorted(results, key=lambda item: item.final_score, reverse=True)
    return [replace(result, rank=idx + 1) for idx, result in enumerate(ranked)]


def _resolve_extraction_payload(
    jd_text: str,
    resumes: list[ResumeDocument],
    use_hybrid_score: bool,
    use_structured_extraction_score: bool,
    use_llm_extraction: bool,
    structured_extraction: dict[str, Any] | None,
    skill_extractor: SkillExtractor | None,
) -> dict[str, Any]:
    if not use_hybrid_score or not use_structured_extraction_score:
        return {}
    if structured_extraction:
        return structured_extraction
    return extract_screening_json(
        jd_text=jd_text,
        resumes=resumes,
        use_llm=use_llm_extraction,
        skill_extractor=skill_extractor,
    )


def _resume_extractions_by_identity(extraction_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not extraction_payload:
        return {}
    rows = extraction_payload.get("resumes", [])
    if isinstance(rows, dict):
        indexed: dict[str, dict[str, Any]] = {}
        for key, value in rows.items():
            if not isinstance(value, dict):
                continue
            extraction = value.get("extraction", value)
            if isinstance(extraction, dict):
                indexed[str(key)] = extraction
        return indexed
    if not isinstance(rows, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        extraction = row.get("extraction")
        if not isinstance(extraction, dict):
            continue
        for key in [row.get("candidate_id"), row.get("filename")]:
            if key:
                indexed[str(key)] = extraction
    return indexed
