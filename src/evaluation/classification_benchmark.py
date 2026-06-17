from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config.settings import EVALUATION_OUTPUT_DIR, PROCESSED_DATA_DIR
from src.evaluation.classification_metrics import (
    evaluate_binary_classification,
    classification_report_dataframe,
)
from src.evaluation.label_utils import convert_label_to_binary, find_label_column
from src.services.matching_service import get_matcher


DEFAULT_SCORE_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]


def validate_pair_dataset(
    pairs: pd.DataFrame,
    label_column: str,
    label_threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a compact dataset validation table for the benchmark output."""
    y_true = pairs[label_column].map(
        lambda value: convert_label_to_binary(value, label_threshold)
    )
    rows: list[dict[str, object]] = [
        {"metric": "num_pairs", "value": len(pairs)},
        {"metric": "num_columns", "value": len(pairs.columns)},
        {"metric": "label_column", "value": label_column},
        {"metric": "label_threshold", "value": label_threshold},
        {"metric": "label_positive_count", "value": int(y_true.sum())},
        {"metric": "label_negative_count", "value": int(len(y_true) - y_true.sum())},
        {"metric": "label_positive_rate", "value": round(float(y_true.mean()), 6)},
        {"metric": "duplicate_full_rows", "value": int(pairs.duplicated().sum())},
    ]

    if {"resume_id", "job_id"}.issubset(pairs.columns):
        rows.append(
            {
                "metric": "duplicate_resume_job_pairs",
                "value": int(pairs.duplicated(subset=["resume_id", "job_id"]).sum()),
            }
        )

    if {"resume_text", "job_description"}.issubset(pairs.columns):
        rows.append(
            {
                "metric": "duplicate_text_pairs",
                "value": int(
                    pairs.duplicated(subset=["resume_text", "job_description"]).sum()
                ),
            }
        )

    for col in ["resume_id", "job_id"]:
        if col in pairs.columns:
            rows.extend(
                [
                    {
                        "metric": f"{col}_unique_count",
                        "value": int(pairs[col].nunique(dropna=False)),
                    },
                    {"metric": f"{col}_missing_count", "value": int(pairs[col].isna().sum())},
                ]
            )

    for col in ["resume_text", "job_description"]:
        if col not in pairs.columns:
            continue
        text = pairs[col].fillna("").astype(str).str.strip()
        lengths = text.str.len()
        rows.extend(
            [
                {"metric": f"{col}_missing_count", "value": int(pairs[col].isna().sum())},
                {"metric": f"{col}_empty_count", "value": int((text == "").sum())},
                {"metric": f"{col}_min_chars", "value": int(lengths.min())},
                {"metric": f"{col}_median_chars", "value": float(lengths.median())},
                {"metric": f"{col}_max_chars", "value": int(lengths.max())},
            ]
        )

    if "job_id" in pairs.columns:
        group_sizes = pairs.groupby("job_id").size()
        label_frame = pd.DataFrame(
            {"job_id": pairs["job_id"], "label_binary": y_true}
        )
        labels_by_job = label_frame.groupby("job_id")["label_binary"]
        rows.extend(
            [
                {"metric": "job_count", "value": int(group_sizes.size)},
                {"metric": "job_group_min_pairs", "value": int(group_sizes.min())},
                {"metric": "job_group_median_pairs", "value": float(group_sizes.median())},
                {"metric": "job_group_max_pairs", "value": int(group_sizes.max())},
                {
                    "metric": "jobs_with_positive_labels",
                    "value": int(labels_by_job.sum().gt(0).sum()),
                },
                {
                    "metric": "jobs_with_no_positive_labels",
                    "value": int(labels_by_job.sum().eq(0).sum()),
                },
                {
                    "metric": "jobs_with_all_positive_labels",
                    "value": int(labels_by_job.mean().eq(1.0).sum()),
                },
            ]
        )

    return pd.DataFrame(rows)


def fit_unsupervised_matcher(model_key: str, pairs: pd.DataFrame):
    """Fit unsupervised corpus state when a matcher supports it."""
    matcher = get_matcher(model_key)
    if model_key.lower() == "tfidf" and hasattr(matcher, "fit"):
        corpus = pd.concat(
            [pairs["job_description"], pairs["resume_text"]],
            ignore_index=True,
        )
        matcher.fit(corpus.dropna().astype(str).drop_duplicates().tolist())
    return matcher


def score_pairs_for_model(
    pairs: pd.DataFrame,
    model_key: str,
    limit_pairs: int | None = None,
    fit_pairs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Score each CV-JD pair independently.

    No groupby job_id is used here.
    Each row is treated as one independent classification sample:
    resume_text + job_description -> matching score.
    """
    if limit_pairs is not None:
        pairs = pairs.head(limit_pairs).copy()
    else:
        pairs = pairs.copy()

    matcher = fit_unsupervised_matcher(
        model_key,
        fit_pairs if fit_pairs is not None else pairs,
    )

    scores: list[float] = []

    started = time.perf_counter()

    for row in pairs.itertuples(index=False):
        jd_text = str(getattr(row, "job_description"))
        resume_text = str(getattr(row, "resume_text"))

        # matcher.score expects:
        # 1 job description + list of resume texts
        # Because this is pair classification, we pass only one resume.
        pair_score = matcher.score(jd_text, [resume_text])

        if not pair_score:
            scores.append(0.0)
        else:
            scores.append(float(pair_score[0]))

    elapsed = time.perf_counter() - started

    scored_pairs = pairs.copy()
    scored_pairs["score"] = scores
    scored_pairs["model"] = model_key

    avg_runtime = elapsed / max(len(scored_pairs), 1)

    return scored_pairs, avg_runtime


def score_distribution_dataframe(
    scored_pairs: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Summarize score and prediction distributions for debugging."""
    scores = pd.to_numeric(scored_pairs["score"], errors="coerce")
    percentiles = scores.quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    rows: list[dict[str, object]] = [
        {"metric": "score_count", "value": int(scores.count())},
        {"metric": "score_missing_count", "value": int(scores.isna().sum())},
        {"metric": "score_unique_count", "value": int(scores.nunique(dropna=False))},
        {"metric": "score_mean", "value": float(scores.mean())},
        {"metric": "score_std", "value": float(scores.std(ddof=0))},
        {"metric": "score_min", "value": float(scores.min())},
        {"metric": "score_max", "value": float(scores.max())},
    ]
    rows.extend(
        {"metric": f"score_p{int(q * 100):02d}", "value": float(value)}
        for q, value in percentiles.items()
    )

    filled_scores = scores.fillna(0.0)
    for threshold in thresholds:
        predictions = (filled_scores >= threshold).astype(int)
        rows.extend(
            [
                {
                    "metric": f"predicted_positive_at_{threshold:g}",
                    "value": int(predictions.sum()),
                },
                {
                    "metric": f"predicted_negative_at_{threshold:g}",
                    "value": int(len(predictions) - predictions.sum()),
                },
                {
                    "metric": f"predicted_positive_rate_at_{threshold:g}",
                    "value": float(predictions.mean()),
                },
            ]
        )

    return pd.DataFrame(rows)


def threshold_sweep_dataframe(
    y_true: list[int],
    y_score: list[float],
    thresholds: list[float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for score_threshold in thresholds:
        metrics = evaluate_binary_classification(
            y_true=y_true,
            y_score=y_score,
            threshold=score_threshold,
        )
        report_df = classification_report_dataframe(metrics)
        report_df["score_threshold"] = score_threshold
        rows.append(report_df)
    return pd.concat(rows, ignore_index=True)


def run_classification_benchmark(
    pairs_path: Path,
    model_keys: list[str],
    threshold: float | None = None,
    output_dir: Path = EVALUATION_OUTPUT_DIR,
    limit_pairs: int | None = None,
    label_threshold: float = 0.5,
    score_thresholds: list[float] | None = None,
    fit_pairs_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run classification metrics for resume-JD matching.

    This version evaluates each row independently:
    each CV-JD pair is classified as match or not match.
    """
    pairs = pd.read_csv(pairs_path)

    # Clean column names to avoid hidden spaces
    pairs.columns = pairs.columns.str.strip()
    fit_pairs = pd.read_csv(fit_pairs_path) if fit_pairs_path else None
    if fit_pairs is not None:
        fit_pairs.columns = fit_pairs.columns.str.strip()

    required_columns = ["resume_text", "job_description"]

    for col in required_columns:
        if col not in pairs.columns:
            raise KeyError(
                f"Missing required column: {col}. "
                f"Available columns: {pairs.columns.tolist()}"
            )
        if fit_pairs is not None and col not in fit_pairs.columns:
            raise KeyError(
                f"Missing required column in --fit-pairs file: {col}. "
                f"Available columns: {fit_pairs.columns.tolist()}"
            )

    label_column = find_label_column(pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = score_thresholds or (
        [threshold] if threshold is not None else DEFAULT_SCORE_THRESHOLDS
    )
    thresholds = sorted({float(value) for value in thresholds})
    diagnostics = validate_pair_dataset(
        pairs=pairs,
        label_column=label_column,
        label_threshold=label_threshold,
    )
    diagnostics.to_csv(output_dir / "classification_dataset_diagnostics.csv", index=False)

    summary_tables: list[pd.DataFrame] = []

    for model_key in model_keys:
        scored_pairs, avg_runtime = score_pairs_for_model(
            pairs=pairs,
            model_key=model_key,
            limit_pairs=limit_pairs,
            fit_pairs=fit_pairs,
        )

        scored_path = output_dir / f"scored_pairs_classification_{model_key}.csv"
        scored_pairs.to_csv(scored_path, index=False)

        # Ground-truth labels from dataset
        y_true = [
            convert_label_to_binary(label, label_threshold=label_threshold)
            for label in scored_pairs[label_column]
        ]

        # Model scores
        y_score = scored_pairs["score"].tolist()

        score_distribution_dataframe(scored_pairs, thresholds).to_csv(
            output_dir / f"classification_score_distribution_{model_key}.csv",
            index=False,
        )

        report_df = threshold_sweep_dataframe(
            y_true=y_true,
            y_score=y_score,
            thresholds=thresholds,
        )
        report_df["model"] = model_key
        report_df["avg_runtime_seconds_per_pair"] = avg_runtime
        report_df["label_threshold"] = label_threshold
        report_df["num_pairs"] = len(scored_pairs)

        summary_tables.append(report_df)

        metrics_path = output_dir / f"classification_metrics_{model_key}.csv"
        report_df.to_csv(metrics_path, index=False)

    comparison = pd.concat(summary_tables, ignore_index=True)
    comparison.to_csv(output_dir / "classification_comparison.csv", index=False)

    return comparison


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("At least one score threshold is required.")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate resume-JD matching models with classification metrics."
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        default=PROCESSED_DATA_DIR / "test_pairs.csv",
    )

    parser.add_argument(
        "--models",
        default="tfidf,bm25",
        help="Comma-separated model keys: tfidf,bm25,word2vec,glove,sbert,e5,bge,google",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Single model-score threshold. If omitted, the default threshold sweep is used.",
    )

    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated model-score thresholds, e.g. 0.1,0.2,0.3,0.5,0.7.",
    )

    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Threshold for converting numeric relevance labels to binary ground truth.",
    )

    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="Limit number of CV-JD pairs for quick experiments.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_OUTPUT_DIR,
    )

    parser.add_argument(
        "--fit-pairs",
        type=Path,
        default=None,
        help="Optional train/validation pair CSV used to fit unsupervised model state such as TF-IDF IDF weights.",
    )

    args = parser.parse_args()

    model_keys = [
        model.strip()
        for model in args.models.split(",")
        if model.strip()
    ]
    score_thresholds = (
        parse_thresholds(args.thresholds)
        if args.thresholds
        else ([args.threshold] if args.threshold is not None else DEFAULT_SCORE_THRESHOLDS)
    )

    comparison = run_classification_benchmark(
        pairs_path=args.pairs,
        model_keys=model_keys,
        threshold=args.threshold,
        output_dir=args.output_dir,
        limit_pairs=args.limit_pairs,
        label_threshold=args.label_threshold,
        score_thresholds=score_thresholds,
        fit_pairs_path=args.fit_pairs,
    )

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
