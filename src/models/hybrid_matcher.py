from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import DEFAULT_HYBRID_WEIGHTS
from src.preprocessing.skill_extractor import SkillAnalysis, SkillExtractor
from src.preprocessing.text_cleaner import clean_for_matching
from src.utils.constants import CERTIFICATION_KEYWORDS, DEGREE_KEYWORDS, EXPERIENCE_PATTERNS


@dataclass
class HybridScoreBreakdown:
    semantic_score: float
    skill_score: float
    experience_score: float
    education_score: float
    final_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    resume_skills: list[str]
    jd_skills: list[str]
    detected_resume_years: float
    required_years: float
    education_signals: list[str]
    certification_signals: list[str]


class HybridMatcher:
    def __init__(
        self,
        skill_extractor: SkillExtractor | None = None,
        weights: dict[str, float] | None = None,
    ):
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.weights = self._normalize_weights(weights or DEFAULT_HYBRID_WEIGHTS)

    def explainable_score(self, semantic_score: float, resume_text: str, jd_text: str) -> HybridScoreBreakdown:
        skill_analysis = self.skill_extractor.analyze(resume_text, jd_text)
        required_years = extract_required_years(jd_text)
        resume_years = extract_resume_years(resume_text)
        exp_score = experience_match_score(resume_years, required_years)
        edu_score, edu_signals, cert_signals = education_certification_score(resume_text, jd_text)

        semantic = clamp01(semantic_score)
        final = (
            self.weights["semantic"] * semantic
            + self.weights["skills"] * skill_analysis.skill_score
            + self.weights["experience"] * exp_score
            + self.weights["education"] * edu_score
        )
        return HybridScoreBreakdown(
            semantic_score=semantic,
            skill_score=skill_analysis.skill_score,
            experience_score=exp_score,
            education_score=edu_score,
            final_score=float(round(clamp01(final) * 100.0, 2)),
            matched_skills=skill_analysis.matched_skills,
            missing_skills=skill_analysis.missing_skills,
            resume_skills=skill_analysis.resume_skills,
            jd_skills=skill_analysis.jd_skills,
            detected_resume_years=resume_years,
            required_years=required_years,
            education_signals=edu_signals,
            certification_signals=cert_signals,
        )

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        expected = ["semantic", "skills", "experience", "education"]
        values = {key: max(0.0, float(weights.get(key, 0.0))) for key in expected}
        total = sum(values.values())
        if total == 0:
            return DEFAULT_HYBRID_WEIGHTS.copy()
        return {key: value / total for key, value in values.items()}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def extract_required_years(jd_text: str) -> float:
    return _extract_max_years(jd_text)


def extract_resume_years(resume_text: str) -> float:
    return _extract_max_years(resume_text)


def _extract_max_years(text: str) -> float:
    cleaned = clean_for_matching(text)
    values: list[float] = []
    for pattern in EXPERIENCE_PATTERNS:
        for match in re.finditer(pattern, cleaned):
            try:
                values.append(float(match.group("years")))
            except (ValueError, IndexError):
                continue
    return max(values) if values else 0.0


def experience_match_score(resume_years: float, required_years: float) -> float:
    if required_years <= 0:
        return 1.0 if resume_years > 0 else 0.5
    return clamp01(resume_years / required_years)


def education_certification_score(resume_text: str, jd_text: str) -> tuple[float, list[str], list[str]]:
    resume = clean_for_matching(resume_text)
    jd = clean_for_matching(jd_text)

    jd_education = [kw for kw in DEGREE_KEYWORDS if kw in jd]
    jd_certs = [kw for kw in CERTIFICATION_KEYWORDS if kw in jd]
    resume_education = [kw for kw in DEGREE_KEYWORDS if kw in resume]
    resume_certs = [kw for kw in CERTIFICATION_KEYWORDS if kw in resume]

    required_signals = jd_education + jd_certs
    if not required_signals:
        observed = bool(resume_education or resume_certs)
        return (1.0 if observed else 0.5), resume_education, resume_certs

    matched = len(set(required_signals).intersection(resume_education + resume_certs))
    score = matched / len(set(required_signals))
    return clamp01(score), resume_education, resume_certs


def recommendation_label(final_score: float) -> str:
    if final_score >= 75.0:
        return "Strong Match"
    if final_score >= 55.0:
        return "Potential Match"
    return "Low Match"
