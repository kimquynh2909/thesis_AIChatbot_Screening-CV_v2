from __future__ import annotations

from io import StringIO

import pandas as pd

from src.services.matching_service import MatchResult


def results_to_dataframe(results: list[MatchResult]) -> pd.DataFrame:
    return pd.DataFrame([result.to_export_dict() for result in results])


def results_to_csv_bytes(results: list[MatchResult]) -> bytes:
    dataframe = results_to_dataframe(results)
    return dataframe.to_csv(index=False).encode("utf-8")


def screening_report_markdown(jd_text: str, results: list[MatchResult]) -> str:
    output = StringIO()
    output.write("# HR Screening Summary\n\n")
    output.write("This report supports HR screening and does not make final hiring decisions.\n\n")
    output.write("## Job Description Preview\n\n")
    output.write(jd_text[:1500].strip())
    if len(jd_text) > 1500:
        output.write("...")
    output.write("\n\n## Ranked Candidates\n\n")
    for result in results:
        output.write(f"### #{result.rank} {result.filename}\n\n")
        output.write(f"- Final score: {result.final_score:.2f}%\n")
        output.write(f"- Semantic score: {result.semantic_score:.2f}%\n")
        output.write(f"- Recommendation: {result.recommendation}\n")
        output.write(f"- Matched skills: {', '.join(result.matched_skills) or 'None detected'}\n")
        output.write(f"- Missing skills: {', '.join(result.missing_skills) or 'None detected'}\n")
        output.write(f"- Detected experience years: {result.detected_resume_years:g}\n")
        output.write(f"- Required experience years: {result.required_years:g}\n\n")
    return output.getvalue()
