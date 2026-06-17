from __future__ import annotations

import pandas as pd


LABEL_COLUMN_CANDIDATES = [
    "label",
    "lable",
    "relevance",
    "Best Match",
    "best_match",
    "matched_score",
    "score_label",
]


def find_label_column(df: pd.DataFrame) -> str:
    """Find the ground-truth label or relevance column in a pair dataset."""
    for col in LABEL_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    raise KeyError(
        f"No label column found. Expected one of {LABEL_COLUMN_CANDIDATES}. "
        f"Available columns: {df.columns.tolist()}"
    )


def label_to_relevance_score(value: object) -> float:
    """Normalize common binary/string/percentage labels into a 0-1 relevance score."""
    if pd.isna(value):
        return 0.0

    value_str = str(value).strip().lower()

    positive_labels = {
        "best match",
        "good match",
        "strong match",
        "match",
        "matched",
        "relevant",
        "yes",
        "true",
    }

    negative_labels = {
        "not match",
        "no match",
        "poor match",
        "bad match",
        "irrelevant",
        "no",
        "false",
    }

    graded_labels = {
        "medium": 0.66,
        "potential": 0.66,
        "partial": 0.5,
        "low": 0.25,
    }

    if value_str in positive_labels:
        return 1.0
    if value_str in negative_labels:
        return 0.0
    if value_str in graded_labels:
        return graded_labels[value_str]

    try:
        numeric_value = float(value_str.replace("%", ""))
    except ValueError as exc:
        raise ValueError(f"Cannot convert label value to relevance score: {value}") from exc

    if numeric_value > 1.0:
        numeric_value = numeric_value / 100.0

    return max(0.0, min(1.0, numeric_value))


def convert_label_to_binary(value: object, label_threshold: float = 0.5) -> int:
    """
    Convert dataset label to binary ground truth.

    This threshold is only for labels/relevance scores. It must be kept separate
    from the model-score threshold used to create predictions.
    """
    return int(label_to_relevance_score(value) >= label_threshold)
