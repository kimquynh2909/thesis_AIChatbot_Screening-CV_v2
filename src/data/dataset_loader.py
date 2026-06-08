from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import RAW_DATA_DIR
from src.preprocessing.text_cleaner import clean_for_display
from src.utils.constants import RELEVANCE_LABEL_THRESHOLD


RESUME_COLUMNS = [
    "resume",
    "Resume",
    "Resume_str",
    "Resume_Text",
    "resume_text",
    "career_objective",
    "skills",
    "Skills",
    "responsibilities",
    "Responsibilities",
    "professional_company_names",
    "positions",
    "education",
    "Education",
]

JD_COLUMNS = [
    "job_description",
    "Job Description",
    "Description",
    "description",
    "JD",
    "job_desc",
    "responsibilities_required",
]

SCORE_COLUMNS = [
    "matched_score",
    "match_score",
    "Best Match",
    "best_match",
    "score",
    "relevance",
    "label",
]

JOB_ID_COLUMNS = ["job_id", "Job Id", "JobID", "jobid", "job_role", "Job Roles", "Title", "Job Title"]
RESUME_ID_COLUMNS = ["resume_id", "ResumeID", "ID", "candidate_id", "Job Applicant Name", "Name"]
CATEGORY_COLUMNS = ["Category", "category", "job_role", "Job_Role", "Role", "role"]


def discover_data_files(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    patterns = ["*.csv", "*.json", "*.jsonl", "*.xlsx"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_dir.rglob(pattern))
    return sorted(files)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(records)
    raise ValueError(f"Unsupported dataset file type: {path.suffix}")


def find_first_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def compose_text(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        if column in row.index and pd.notna(row[column]):
            value = row[column]
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            parts.append(str(value))
    return clean_for_display("\n".join(parts))


def load_primary_ranking_pairs(path: Path) -> pd.DataFrame:
    """Load paired resume-JD rows from a dataset with a match or relevance score."""
    df = read_table(path)
    resume_id_col = find_first_column(df.columns, RESUME_ID_COLUMNS)
    job_id_col = find_first_column(df.columns, JOB_ID_COLUMNS)
    score_col = find_first_column(df.columns, SCORE_COLUMNS)
    jd_col = find_first_column(df.columns, JD_COLUMNS)

    if not score_col:
        raise ValueError("Could not find a match/relevance score column in the selected file.")

    resume_text_columns = [column for column in RESUME_COLUMNS if column in df.columns]
    jd_text_columns = [jd_col] if jd_col else [column for column in JD_COLUMNS if column in df.columns]
    if not resume_text_columns:
        raise ValueError("Could not find resume text columns in the selected file.")
    if not jd_text_columns:
        raise ValueError("Could not find job description text columns in the selected file.")

    output = pd.DataFrame()
    output["pair_id"] = [f"pair_{idx}" for idx in range(len(df))]
    output["resume_text"] = df.apply(lambda row: compose_text(row, resume_text_columns), axis=1)
    output["job_description"] = df.apply(lambda row: compose_text(row, jd_text_columns), axis=1)
    output["resume_id"] = df[resume_id_col].astype(str) if resume_id_col else [f"resume_{idx}" for idx in range(len(df))]
    if job_id_col:
        output["job_id"] = df[job_id_col].astype(str)
    else:
        output["job_id"] = output["job_description"].map(lambda text: "job_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12])
    output["relevance"] = df[score_col].apply(normalize_relevance_score)
    output["label"] = (output["relevance"] >= RELEVANCE_LABEL_THRESHOLD).astype(int)
    output["source_file"] = str(path)
    return output


def normalize_relevance_score(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {
            "yes": 1.0,
            "true": 1.0,
            "match": 1.0,
            "best match": 1.0,
            "strong": 1.0,
            "medium": 0.66,
            "potential": 0.66,
            "partial": 0.5,
            "low": 0.25,
            "no": 0.0,
            "false": 0.0,
            "not match": 0.0,
        }
        if lowered in mapping:
            return mapping[lowered]
        value = lowered.replace("%", "")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def load_resume_category_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path)
    text_col = find_first_column(df.columns, ["Resume_str", "Resume", "Resume_Text", "resume_text", "cleaned_text"])
    category_col = find_first_column(df.columns, CATEGORY_COLUMNS)
    id_col = find_first_column(df.columns, RESUME_ID_COLUMNS)
    if not text_col or not category_col:
        raise ValueError("Resume dataset must include resume text and category columns.")
    output = pd.DataFrame()
    output["resume_id"] = df[id_col].astype(str) if id_col else [f"resume_{idx}" for idx in range(len(df))]
    output["resume_text"] = df[text_col].astype(str).map(clean_for_display)
    output["resume_category"] = df[category_col].astype(str)
    return output


def load_job_description_dataset(path: Path) -> pd.DataFrame:
    df = read_table(path)
    description_col = find_first_column(df.columns, JD_COLUMNS)
    title_col = find_first_column(df.columns, ["Title", "Job Title", "job_title", "Role", "role"])
    skills_col = find_first_column(df.columns, ["Skills", "skills", "Keywords", "job_skill_set"])
    id_col = find_first_column(df.columns, JOB_ID_COLUMNS)
    if not description_col:
        raise ValueError("Job dataset must include a job description column.")
    output = pd.DataFrame()
    output["job_id"] = df[id_col].astype(str) if id_col else [f"job_{idx}" for idx in range(len(df))]
    output["job_title"] = df[title_col].astype(str) if title_col else "Unknown"
    if skills_col:
        output["job_description"] = (df[title_col].astype(str) + "\n" if title_col else "") + df[description_col].astype(str) + "\nSkills: " + df[skills_col].astype(str)
    else:
        output["job_description"] = df[description_col].astype(str)
    output["job_description"] = output["job_description"].map(clean_for_display)
    return output
