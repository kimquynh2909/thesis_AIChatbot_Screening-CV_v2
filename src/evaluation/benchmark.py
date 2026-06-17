from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config.settings import EVALUATION_OUTPUT_DIR, PROCESSED_DATA_DIR
from src.evaluation.label_utils import convert_label_to_binary, find_label_column
from src.evaluation.ranking_metrics import aggregate_metric_table, evaluate_ranked_groups
from src.services.matching_service import ResumeDocument, get_matcher


def score_pairs_for_model(pairs: pd.DataFrame, model_key: str, limit_jobs: int | None = None) -> tuple[pd.DataFrame, float]:
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


def run_benchmark(
    pairs_path: Path,
    model_keys: list[str],
    output_dir: Path = EVALUATION_OUTPUT_DIR,
    limit_jobs: int | None = None,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tables: list[pd.DataFrame] = []

    for model_key in model_keys:
        scored_pairs, avg_runtime = score_pairs_for_model(pairs, model_key, limit_jobs=limit_jobs)
        scored_path = output_dir / f"scored_pairs_{model_key}.csv"
        scored_pairs.to_csv(scored_path, index=False)
        if "relevance" not in scored_pairs.columns:
            label_column = find_label_column(scored_pairs)
            scored_pairs["relevance"] = scored_pairs[label_column].map(convert_label_to_binary)
        group_metrics = evaluate_ranked_groups(scored_pairs)
        metrics_path = output_dir / f"group_metrics_{model_key}.csv"
        group_metrics.to_csv(metrics_path, index=False)
        summary = aggregate_metric_table(group_metrics, model_key)
        summary["avg_runtime_seconds_per_pair"] = avg_runtime
        summary_tables.append(summary)

    comparison = pd.concat(summary_tables, ignore_index=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark resume-JD matching models.")
    parser.add_argument("--pairs", type=Path, default=PROCESSED_DATA_DIR / "test_pairs.csv")
    parser.add_argument(
        "--models",
        default="tfidf,bm25",
        help="Comma-separated model keys: tfidf,bm25,word2vec,glove,sbert,e5,bge,google",
    )
    parser.add_argument("--limit-jobs", type=int, default=None, help="Limit jobs for quick experiments.")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_OUTPUT_DIR)
    args = parser.parse_args()

    model_keys = [model.strip() for model in args.models.split(",") if model.strip()]
    comparison = run_benchmark(args.pairs, model_keys, args.output_dir, args.limit_jobs)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
