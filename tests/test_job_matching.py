from __future__ import annotations

from src.services.matching_service import JobDescriptionDocument, ResumeDocument, match_jobs_to_resume


RESUME = """Machine learning engineer with 4 years of experience.
Skills: Python, SQL, PyTorch, natural language processing, Docker, AWS."""

GOOD_JOB = "Machine learning engineer required with Python, SQL, PyTorch, NLP, Docker, AWS and 3 years of experience."

WEAK_JOB = "Frontend developer with React, CSS, JavaScript and UI design experience."


def test_matching_service_ranks_job_descriptions_for_one_resume() -> None:
    results = match_jobs_to_resume(
        ResumeDocument("resume", "resume.txt", RESUME),
        [
            JobDescriptionDocument("good_job", "ml_engineer.txt", GOOD_JOB),
            JobDescriptionDocument("weak_job", "frontend.txt", WEAK_JOB),
        ],
        model_key="tfidf",
        use_llm_extraction=False,
    )

    assert results[0].candidate_id == "good_job"
    assert results[0].rank == 1
    assert results[0].final_score > results[1].final_score


def test_matching_service_uses_structured_json_for_job_matching_score() -> None:
    payload = {
        "job_descriptions": [
            {
                "job_id": "kubernetes_job",
                "filename": "kubernetes.txt",
                "job_description": {
                    "required_skills": ["Kubernetes"],
                    "preferred_skills": [],
                    "tools": [],
                    "education": [],
                    "experience": [],
                },
                "resume": {
                    "skills": ["Kubernetes"],
                    "education": [],
                    "experience": [],
                    "projects": [],
                },
            }
        ]
    }

    results = match_jobs_to_resume(
        ResumeDocument("resume", "resume.txt", "Frontend developer with React experience."),
        [JobDescriptionDocument("kubernetes_job", "kubernetes.txt", "Need Kubernetes.")],
        model_key="tfidf",
        weights={"semantic": 0.0, "skills": 1.0, "experience": 0.0, "education": 0.0},
        structured_extraction=payload,
    )

    assert results[0].matched_skills == ["kubernetes"]
    assert results[0].missing_skills == []
    assert results[0].final_score == 100.0
