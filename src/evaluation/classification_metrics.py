from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_binary_classification(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, object]:
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true and y_score must have the same length. Got {len(y_true)} and {len(y_score)}.")

    y_pred = [1 if score >= threshold else 0 for score in y_score]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "actual_negative": int(sum(1 for label in y_true if label == 0)),
        "actual_positive": int(sum(1 for label in y_true if label == 1)),
        "predicted_negative": int(sum(1 for label in y_pred if label == 0)),
        "predicted_positive": int(sum(1 for label in y_pred if label == 1)),
        "confusion_matrix": matrix,
    }
    return metrics


def classification_report_dataframe(metrics: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{key: value for key, value in metrics.items() if key != "confusion_matrix"}])
