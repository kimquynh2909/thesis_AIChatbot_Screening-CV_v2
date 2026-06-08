from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_binary_classification(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, object]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None
    return metrics


def classification_report_dataframe(metrics: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{key: value for key, value in metrics.items() if key != "confusion_matrix"}])
