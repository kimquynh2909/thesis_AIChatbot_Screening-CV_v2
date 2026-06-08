from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def binary_relevance(relevance: Iterable[float], threshold: float = 0.5) -> list[int]:
    return [1 if float(value) >= threshold else 0 for value in relevance]


def precision_at_k(relevance: Iterable[float], k: int, threshold: float = 0.5) -> float:
    labels = binary_relevance(list(relevance)[:k], threshold)
    if k <= 0:
        return 0.0
    return float(sum(labels) / k)


def recall_at_k(relevance: Iterable[float], k: int, total_relevant: int | None = None, threshold: float = 0.5) -> float:
    relevance_list = list(relevance)
    labels = binary_relevance(relevance_list[:k], threshold)
    denominator = total_relevant if total_relevant is not None else sum(binary_relevance(relevance_list, threshold))
    if denominator <= 0:
        return 0.0
    return float(sum(labels) / denominator)


def f1_at_k(relevance: Iterable[float], k: int, threshold: float = 0.5) -> float:
    relevance_list = list(relevance)
    p = precision_at_k(relevance_list, k, threshold)
    r = recall_at_k(relevance_list, k, threshold=threshold)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def dcg_at_k(relevance: Iterable[float], k: int) -> float:
    values = list(map(float, relevance))[:k]
    return float(sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(values)))


def ndcg_at_k(relevance: Iterable[float], k: int) -> float:
    values = list(map(float, relevance))
    ideal = sorted(values, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return float(dcg_at_k(values, k) / ideal_dcg)


def reciprocal_rank(relevance: Iterable[float], threshold: float = 0.5) -> float:
    for idx, label in enumerate(binary_relevance(relevance, threshold), start=1):
        if label:
            return float(1.0 / idx)
    return 0.0


def average_precision(relevance: Iterable[float], threshold: float = 0.5) -> float:
    labels = binary_relevance(relevance, threshold)
    total_relevant = sum(labels)
    if total_relevant == 0:
        return 0.0
    precisions = []
    found = 0
    for idx, label in enumerate(labels, start=1):
        if label:
            found += 1
            precisions.append(found / idx)
    return float(sum(precisions) / total_relevant)


def evaluate_ranked_groups(scored_pairs: pd.DataFrame, k_values: list[int] | None = None) -> pd.DataFrame:
    """Evaluate a scored pair table with columns job_id, score, and relevance."""
    k_values = k_values or [1, 3, 5, 10]
    rows: list[dict[str, float | str | int]] = []
    for job_id, group in scored_pairs.groupby("job_id"):
        ranked = group.sort_values("score", ascending=False)
        relevance = ranked["relevance"].astype(float).tolist()
        row: dict[str, float | str | int] = {
            "job_id": job_id,
            "candidate_count": len(ranked),
            "relevant_count": int(sum(binary_relevance(relevance))),
            "mrr": reciprocal_rank(relevance),
            "map": average_precision(relevance),
        }
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(relevance, k)
            row[f"recall@{k}"] = recall_at_k(relevance, k)
            row[f"f1@{k}"] = f1_at_k(relevance, k)
            row[f"ndcg@{k}"] = ndcg_at_k(relevance, k)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metric_table(group_metrics: pd.DataFrame, model_name: str) -> pd.DataFrame:
    numeric = group_metrics.select_dtypes(include=[np.number])
    means = numeric.mean().to_dict()
    means["model"] = model_name
    return pd.DataFrame([means])
