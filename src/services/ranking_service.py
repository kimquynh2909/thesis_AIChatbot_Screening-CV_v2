from __future__ import annotations

from src.services.matching_service import MatchResult


def filter_candidates_by_skills(results: list[MatchResult], required_skills: list[str]) -> list[MatchResult]:
    required = {skill.lower().strip() for skill in required_skills if skill.strip()}
    if not required:
        return results
    filtered = []
    for result in results:
        resume_skills = {skill.lower() for skill in result.resume_skills}
        if required.issubset(resume_skills):
            filtered.append(result)
    return filtered


def compare_candidates(candidate_a: MatchResult, candidate_b: MatchResult) -> str:
    skill_delta = len(candidate_a.matched_skills) - len(candidate_b.matched_skills)
    score_delta = candidate_a.final_score - candidate_b.final_score
    lines = [
        f"{candidate_a.filename} is ranked #{candidate_a.rank} with {candidate_a.final_score:.2f}%.",
        f"{candidate_b.filename} is ranked #{candidate_b.rank} with {candidate_b.final_score:.2f}%.",
        f"Score difference: {score_delta:.2f} percentage points.",
        f"Matched skill difference: {skill_delta}.",
    ]
    if candidate_a.missing_skills:
        lines.append(f"{candidate_a.filename} missing skills: {', '.join(candidate_a.missing_skills[:8])}.")
    if candidate_b.missing_skills:
        lines.append(f"{candidate_b.filename} missing skills: {', '.join(candidate_b.missing_skills[:8])}.")
    return "\n".join(lines)


def summarize_top_candidates(results: list[MatchResult], top_n: int = 3) -> str:
    if not results:
        return "No screening results are available."
    lines = []
    for result in results[:top_n]:
        matched = ", ".join(result.matched_skills[:8]) or "no dictionary skills detected"
        missing = ", ".join(result.missing_skills[:5]) or "no required dictionary skills missing"
        lines.append(
            f"#{result.rank} {result.filename}: {result.final_score:.2f}% ({result.recommendation}). "
            f"Matched: {matched}. Missing: {missing}."
        )
    return "\n".join(lines)
