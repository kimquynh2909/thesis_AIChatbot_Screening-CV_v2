from __future__ import annotations

from src.llm.structured_extraction import (
    extract_cv_json,
    extract_job_description_json,
    extract_screening_json,
    normalize_cv_extraction,
    normalize_jd_extraction,
    parse_json_object,
)
from src.services.matching_service import ResumeDocument


JD = """Machine Learning Engineer required.
Required skills: Python, SQL, PyTorch, Docker, AWS.
Preferred: LangChain and vector database experience.
Bachelor degree in Computer Science preferred.
Minimum 3 years of machine learning experience.
"""

CV = """Machine learning engineer with 4 years of experience.
Skills: Python, SQL, PyTorch, Docker, AWS, LangChain.
Education: Bachelor Computer Science.
Projects: Built an NLP resume matching project using vector database retrieval.
"""


def test_parse_json_object_accepts_code_fence() -> None:
    parsed = parse_json_object('```json\n{"skills": ["Python"]}\n```')
    assert parsed == {"skills": ["Python"]}


def test_normalize_jd_extraction_keeps_expected_keys() -> None:
    normalized = normalize_jd_extraction({"required_skills": ["Python", "python"], "unknown": ["ignored"]})
    assert list(normalized.keys()) == ["required_skills", "preferred_skills", "tools", "education", "experience"]
    assert normalized["required_skills"] == ["Python"]


def test_normalize_cv_extraction_normalizes_requirement_matches() -> None:
    normalized = normalize_cv_extraction(
        {
            "skills": ["Python"],
            "jd_requirements": [{"requirement": "Python", "status": "yes", "evidence": "Python"}],
        }
    )
    assert normalized["skills"] == ["Python"]
    assert normalized["jd_requirements"] == [{"requirement": "Python", "status": "unclear", "evidence": "Python"}]


def test_deterministic_jd_extraction_returns_calculation_schema() -> None:
    extraction = extract_job_description_json(JD, use_llm=False)
    assert "python" in extraction["required_skills"]
    assert "langchain" in extraction["preferred_skills"]
    assert "3 years" in extraction["experience"]


def test_deterministic_cv_extraction_returns_jd_requirement_evidence() -> None:
    jd_extraction = extract_job_description_json(JD, use_llm=False)
    extraction = extract_cv_json(CV, jd_extraction, use_llm=False)
    assert "python" in extraction["skills"]
    assert extraction["projects"]
    assert any(item["requirement"] == "python" and item["status"] == "matched" for item in extraction["jd_requirements"])


def test_screening_json_wraps_jd_and_resume_extractions() -> None:
    data = extract_screening_json(JD, [ResumeDocument("candidate_1", "resume.txt", CV)], use_llm=False)
    assert set(data.keys()) == {"job_description", "resumes"}
    assert data["resumes"][0]["candidate_id"] == "candidate_1"
    assert data["resumes"][0]["extraction"]["skills"]
