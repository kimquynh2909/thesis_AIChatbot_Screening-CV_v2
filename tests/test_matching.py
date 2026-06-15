from __future__ import annotations

import numpy as np

from src.models.bm25_matcher import BM25Matcher
from src.models.glove_matcher import GloveMatcher
from src.models.tfidf_matcher import TfidfMatcher
from src.models.word2vec_matcher import Word2VecMatcher
from src.services.matching_service import ResumeDocument, get_matcher, match_resumes


JD = "Machine learning engineer required with Python, SQL, PyTorch, NLP, Docker, AWS and 3 years of experience."

GOOD_RESUME = """Machine learning engineer with 4 years of experience.
Skills: Python, SQL, PyTorch, natural language processing, Docker, AWS."""

WEAK_RESUME = """Frontend developer with React, CSS, JavaScript and UI design experience."""

TEST_STATIC_VECTORS = {
    "machine": np.array([1.0, 0.0, 0.0]),
    "learning": np.array([1.0, 0.0, 0.0]),
    "engineer": np.array([1.0, 0.0, 0.0]),
    "python": np.array([1.0, 0.0, 0.0]),
    "sql": np.array([1.0, 0.0, 0.0]),
    "frontend": np.array([0.0, 1.0, 0.0]),
    "react": np.array([0.0, 1.0, 0.0]),
    "css": np.array([0.0, 1.0, 0.0]),
    "javascript": np.array([0.0, 1.0, 0.0]),
}


def test_tfidf_scores_relevant_resume_higher() -> None:
    scores = TfidfMatcher().score(JD, [GOOD_RESUME, WEAK_RESUME])
    assert scores[0] > scores[1]


def test_bm25_scores_relevant_resume_higher() -> None:
    scores = BM25Matcher().score(JD, [GOOD_RESUME, WEAK_RESUME])
    assert scores[0] > scores[1]


def test_word2vec_scores_relevant_resume_higher_with_static_vectors() -> None:
    scores = Word2VecMatcher(vectors=TEST_STATIC_VECTORS).score(JD, [GOOD_RESUME, WEAK_RESUME])
    assert scores[0] > scores[1]


def test_glove_scores_relevant_resume_higher_with_static_vectors() -> None:
    scores = GloveMatcher(vectors=TEST_STATIC_VECTORS).score(JD, [GOOD_RESUME, WEAK_RESUME])
    assert scores[0] > scores[1]


def test_matching_service_dispatches_static_embedding_models() -> None:
    assert isinstance(get_matcher("word2vec"), Word2VecMatcher)
    assert isinstance(get_matcher("glove"), GloveMatcher)


def test_matching_service_ranks_candidates() -> None:
    results = match_resumes(
        JD,
        [
            ResumeDocument("good", "good.txt", GOOD_RESUME),
            ResumeDocument("weak", "weak.txt", WEAK_RESUME),
        ],
        model_key="tfidf",
    )
    assert results[0].candidate_id == "good"
    assert results[0].rank == 1
    assert results[0].final_score > results[1].final_score


def test_matching_service_uses_structured_json_for_hybrid_score() -> None:
    payload = {
        "job_description": {
            "required_skills": ["Kubernetes"],
            "preferred_skills": [],
            "tools": [],
            "education": [],
            "experience": [],
        },
        "resumes": [
            {
                "candidate_id": "json_candidate",
                "filename": "json.txt",
                "extraction": {
                    "skills": ["Kubernetes"],
                    "education": [],
                    "experience": [],
                    "projects": [],
                    "jd_requirements": [
                        {"requirement": "Kubernetes", "status": "matched", "evidence": "LLM extracted Kubernetes evidence"}
                    ],
                },
            }
        ],
    }
    results = match_resumes(
        "Need Kubernetes.",
        [ResumeDocument("json_candidate", "json.txt", "Frontend developer with React experience.")],
        model_key="tfidf",
        weights={"semantic": 0.0, "skills": 1.0, "experience": 0.0, "education": 0.0},
        structured_extraction=payload,
    )
    assert results[0].matched_skills == ["kubernetes"]
    assert results[0].missing_skills == []
    assert results[0].final_score == 100.0
