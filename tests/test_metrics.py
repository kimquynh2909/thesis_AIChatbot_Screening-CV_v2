from __future__ import annotations

import pandas as pd

from src.evaluation.ranking_metrics import (
    average_precision,
    evaluate_ranked_groups,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ranking_metrics_basic_values() -> None:
    relevance = [1, 0, 1, 0]
    assert precision_at_k(relevance, 2) == 0.5
    assert recall_at_k(relevance, 2) == 0.5
    assert reciprocal_rank(relevance) == 1.0
    assert average_precision(relevance) > 0.0
    assert ndcg_at_k(relevance, 3) > 0.0


def test_evaluate_ranked_groups() -> None:
    frame = pd.DataFrame(
        {
            "job_id": ["j1", "j1", "j2", "j2"],
            "resume_id": ["a", "b", "c", "d"],
            "score": [0.9, 0.1, 0.2, 0.8],
            "relevance": [1.0, 0.0, 0.0, 1.0],
        }
    )
    metrics = evaluate_ranked_groups(frame, k_values=[1])
    assert set(metrics["job_id"]) == {"j1", "j2"}
    assert metrics["precision@1"].mean() == 1.0
