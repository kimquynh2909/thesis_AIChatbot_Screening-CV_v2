from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PROCESSED_DATA_DIR, RANDOM_SEED
from src.data.dataset_loader import (
    load_job_description_dataset,
    load_primary_ranking_pairs,
    load_resume_category_dataset,
)
from src.preprocessing.skill_extractor import SkillExtractor


def normalize_role(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def create_weak_pairs(
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    negatives_per_positive: int = 3,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create weak labels when direct CV-JD relevance labels are unavailable."""
    rng = np.random.default_rng(seed)
    extractor = SkillExtractor()
    pairs: list[dict[str, object]] = []

    jobs = jobs.copy()
    jobs["role_norm"] = jobs["job_title"].map(normalize_role)
    resumes = resumes.copy()
    resumes["role_norm"] = resumes["resume_category"].map(normalize_role)

    for _, resume in resumes.iterrows():
        resume_role = resume["role_norm"]
        positive_jobs = jobs[jobs["role_norm"].apply(lambda role: role in resume_role or resume_role in role)]
        if positive_jobs.empty:
            positive_jobs = jobs[jobs["job_description"].str.lower().str.contains(resume_role, regex=False, na=False)]
        if positive_jobs.empty:
            continue

        positive_job = positive_jobs.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        skill_analysis = extractor.analyze(resume["resume_text"], positive_job["job_description"])
        pairs.append(_pair_record(resume, positive_job, 1, max(0.66, skill_analysis.skill_score), "role_or_skill_positive"))

        negative_pool = jobs[jobs["job_id"] != positive_job["job_id"]]
        if negative_pool.empty:
            continue
        sample_size = min(negatives_per_positive, len(negative_pool))
        sampled_negatives = negative_pool.sample(sample_size, random_state=int(rng.integers(0, 1_000_000)))
        for _, negative_job in sampled_negatives.iterrows():
            skill_analysis = extractor.analyze(resume["resume_text"], negative_job["job_description"])
            relevance = min(0.49, skill_analysis.skill_score)
            pairs.append(_pair_record(resume, negative_job, 0, relevance, "weak_negative"))

    output = pd.DataFrame(pairs)
    if output.empty:
        raise ValueError("No weak pairs could be created. Check that resume categories and job titles overlap.")
    output["pair_id"] = [f"pair_{idx}" for idx in range(len(output))]
    return output


def _pair_record(resume: pd.Series, job: pd.Series, label: int, relevance: float, source: str) -> dict[str, object]:
    return {
        "resume_id": resume["resume_id"],
        "job_id": job["job_id"],
        "resume_text": resume["resume_text"],
        "job_description": job["job_description"],
        "label": int(label),
        "relevance": float(relevance),
        "source": source,
    }


def split_by_job_id(pairs: pd.DataFrame, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    job_ids = pairs["job_id"].drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
    if len(job_ids) < 3:
        return {"train": pairs.copy(), "validation": pairs.copy(), "test": pairs.copy()}
    train_end = int(len(job_ids) * 0.70)
    validation_end = int(len(job_ids) * 0.85)
    split_ids = {
        "train": set(job_ids[:train_end]),
        "validation": set(job_ids[train_end:validation_end]),
        "test": set(job_ids[validation_end:]),
    }
    return {name: pairs[pairs["job_id"].isin(ids)].reset_index(drop=True) for name, ids in split_ids.items()}


def save_splits(pairs: pd.DataFrame, output_dir: Path = PROCESSED_DATA_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    pairs_path = output_dir / "candidate_job_pairs.csv"
    pairs.to_csv(pairs_path, index=False)
    paths["all"] = pairs_path
    for split, frame in split_by_job_id(pairs).items():
        path = output_dir / f"{split}_pairs.csv"
        frame.to_csv(path, index=False)
        paths[split] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare candidate-job relevance pairs.")
    parser.add_argument("--primary-pairs", type=Path, help="CSV/JSON/XLSX file with resume, JD, and match score columns.")
    parser.add_argument("--resume-file", type=Path, help="Resume category dataset for weak labels.")
    parser.add_argument("--job-file", type=Path, help="Job description dataset for weak labels.")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    if args.primary_pairs:
        pairs = load_primary_ranking_pairs(args.primary_pairs)
    elif args.resume_file and args.job_file:
        resumes = load_resume_category_dataset(args.resume_file)
        jobs = load_job_description_dataset(args.job_file)
        pairs = create_weak_pairs(resumes, jobs)
    else:
        raise SystemExit("Provide --primary-pairs or both --resume-file and --job-file.")

    paths = save_splits(pairs, args.output_dir)
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
