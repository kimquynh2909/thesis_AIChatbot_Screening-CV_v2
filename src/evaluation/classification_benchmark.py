from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config.settings import EVALUATION_OUTPUT_DIR, PROCESSED_DATA_DIR
from src.evaluation.classification_metrics import evaluate_binary_classification, classification_report_dataframe
from src.services.matching_service import ResumeDocument, get_matcher


def score_pairs_for_model(pairs: pd.DataFrame, model_key: str, limit_jobs: int | None = None) -> tuple[pd.DataFrame, float]:
    """Score pairs using the specified model."""
    matcher = get_matcher(model_key)
    rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    grouped = pairs.groupby("job_id", sort=False)
    if limit_jobs:
        grouped = list(grouped)[:limit_jobs]

    for job_id, group in grouped:
        jd_text = str(group.iloc[0]["job_description"])
        resumes = [
            ResumeDocument(candidate_id=str(row.resume_id), filename=str(row.resume_id), text=str(row.resume_text))
            for row in group.itertuples()
        ]
        scores = matcher.score(jd_text, [resume.text for resume in resumes])
        scored = group.copy()
        scored["score"] = scores
        scored["model"] = model_key
        rows.append(scored)
    elapsed = time.perf_counter() - started
    scored_pairs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    avg_runtime = elapsed / max(len(scored_pairs), 1)
    return scored_pairs, avg_runtime


def run_classification_benchmark(
    pairs_path: Path,
    model_keys: list[str],
    threshold: float = 0.5,
    output_dir: Path = EVALUATION_OUTPUT_DIR,
    limit_jobs: int | None = None,
) -> pd.DataFrame:
    """Run classification metrics for resume-JD matching."""
    pairs = pd.read_csv(pairs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tables: list[pd.DataFrame] = []

    for model_key in model_keys:
        scored_pairs, avg_runtime = score_pairs_for_model(pairs, model_key, limit_jobs=limit_jobs)
        scored_path = output_dir / f"scored_pairs_classification_{model_key}.csv"
        scored_pairs.to_csv(scored_path, index=False)

        # Convert relevance labels to binary (1 if >= threshold, 0 otherwise)
        y_true = [1 if score >= threshold else 0 for score in scored_pairs["relevance"]]
        y_score = scored_pairs["score"].tolist()

        # Evaluate
        metrics = evaluate_binary_classification(y_true, y_score, threshold=0.5)
        report_df = classification_report_dataframe(metrics)
        report_df["model"] = model_key
        report_df["avg_runtime_seconds_per_pair"] = avg_runtime
        report_df["threshold"] = threshold
        report_df["num_pairs"] = len(scored_pairs)
        summary_tables.append(report_df)

        # Save detailed metrics
        metrics_path = output_dir / f"classification_metrics_{model_key}.csv"
        report_df.to_csv(metrics_path, index=False)

    comparison = pd.concat(summary_tables, ignore_index=True)
    comparison.to_csv(output_dir / "classification_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate resume-JD matching models with classification metrics.")
    parser.add_argument("--pairs", type=Path, default=PROCESSED_DATA_DIR / "test_pairs.csv")
    parser.add_argument(
        "--models",
        default="tfidf,bm25",
        help="Comma-separated model keys: tfidf,bm25,word2vec,glove,sbert,e5,bge,google",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Relevance threshold for binary classification (0-1).")
    parser.add_argument("--limit-jobs", type=int, default=None, help="Limit jobs for quick experiments.")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_OUTPUT_DIR)
    args = parser.parse_args()

    model_keys = [model.strip() for model in args.models.split(",") if model.strip()]
    comparison = run_classification_benchmark(args.pairs, model_keys, args.threshold, args.output_dir, args.limit_jobs)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
