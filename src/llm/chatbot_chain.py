from __future__ import annotations

import os
from typing import Any

from src.llm.gemini_utils import DEFAULT_GEMINI_CHAT_MODEL, get_google_api_key, resolve_gemini_chat_model
from src.services.ranking_service import compare_candidates, filter_candidates_by_skills, summarize_top_candidates
from src.utils.file_utils import shorten_text


CHATBOT_TEMPLATE = """You are an HR screening assistant.
Answer only from the supplied screening results, job description, and resume excerpts.
If the evidence is not present, say that the available documents do not provide enough evidence.
Do not make final hiring decisions and do not use protected attributes.

Job description:
{job_description}

Screening results:
{results_summary}

Question:
{question}
"""


JOB_MATCH_CHATBOT_TEMPLATE = """You are a job matching assistant.
Answer only from the supplied resume/CV and ranked job match results.
If the evidence is not present, say that the available documents do not provide enough evidence.
Do not make final hiring decisions and do not use protected attributes.

Resume/CV:
{resume_text}

Job match results:
{results_summary}

Question:
{question}
"""


def answer_hr_question(question: str, results: list[Any], jd_text: str, use_llm: bool = True) -> str:
    if not question.strip():
        return "Please enter a question about the screening results."

    if use_llm:
        answer = _try_llm_answer(question, results, jd_text)
        if answer:
            return answer
    return deterministic_answer(question, results)


def answer_job_match_question(question: str, results: list[Any], resume_text: str, use_llm: bool = True) -> str:
    if not question.strip():
        return "Please enter a question about the job match results."

    if use_llm:
        answer = _try_llm_job_match_answer(question, results, resume_text)
        if answer:
            return answer
    return deterministic_job_match_answer(question, results)


def deterministic_answer(question: str, results: list[Any]) -> str:
    q = question.lower()
    if not results:
        return "No screening results are available yet."

    if "top" in q or "summarize" in q:
        return summarize_top_candidates(results, top_n=3)

    if "higher" in q and "than" in q and len(results) >= 2:
        return compare_candidates(results[0], results[1])

    if "missing" in q:
        lines = []
        for result in results[:5]:
            missing = ", ".join(result.missing_skills[:8]) or "no missing required dictionary skills detected"
            lines.append(f"{result.filename}: {missing}")
        return "\n".join(lines)

    known_skills = sorted({skill for result in results for skill in result.resume_skills})
    requested = [skill for skill in known_skills if skill.lower() in q]
    if requested:
        matches = filter_candidates_by_skills(results, requested)
        if not matches:
            return f"No ranked candidate has all requested skills: {', '.join(requested)}."
        return "\n".join(
            f"#{result.rank} {result.filename}: {result.final_score:.2f}%"
            for result in matches
        )

    return summarize_top_candidates(results, top_n=3)


def deterministic_job_match_answer(question: str, results: list[Any]) -> str:
    q = question.lower()
    if not results:
        return "No job match results are available yet."

    if "top" in q or "summarize" in q or "best" in q:
        lines = []
        for result in results[:3]:
            matched = ", ".join(result.matched_skills[:8]) or "no required dictionary skills detected"
            missing = ", ".join(result.missing_skills[:5]) or "no required dictionary skills missing"
            lines.append(
                f"#{result.rank} {result.filename}: {result.final_score:.2f}% ({result.recommendation}). "
                f"Matched: {matched}. Missing: {missing}."
            )
        return "\n".join(lines)

    if "missing" in q:
        lines = []
        for result in results[:5]:
            missing = ", ".join(result.missing_skills[:8]) or "no missing job-required skills detected"
            lines.append(f"{result.filename}: {missing}")
        return "\n".join(lines)

    if "skill" in q:
        known_skills = sorted({skill for result in results for skill in result.jd_skills})
        requested = [skill for skill in known_skills if skill.lower() in q]
        if requested:
            matches = [result for result in results if set(requested).issubset(set(result.jd_skills))]
            if not matches:
                return f"No ranked job description requires all requested skills: {', '.join(requested)}."
            return "\n".join(f"#{result.rank} {result.filename}: {result.final_score:.2f}%" for result in matches)

    return deterministic_job_match_answer("top", results)


def _try_llm_answer(question: str, results: list[Any], jd_text: str) -> str | None:
    api_key = get_google_api_key()
    if not api_key:
        return None
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None

    result_lines = []
    for result in results[:10]:
        result_lines.append(
            f"Rank {result.rank}: {result.filename}; final={result.final_score:.2f}%; "
            f"semantic={result.semantic_score:.2f}%; label={result.recommendation}; "
            f"matched={', '.join(result.matched_skills[:12]) or 'none'}; "
            f"missing={', '.join(result.missing_skills[:12]) or 'none'}; "
            f"experience={result.detected_resume_years:g}/{result.required_years:g} years."
        )
    prompt = PromptTemplate.from_template(CHATBOT_TEMPLATE)
    model_name = resolve_gemini_chat_model(api_key, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_CHAT_MODEL))
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
    )
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "job_description": shorten_text(jd_text, 4000),
                "results_summary": "\n".join(result_lines),
                "question": question,
            }
        )
        return getattr(response, "content", str(response)).strip()
    except Exception:
        return None


def _try_llm_job_match_answer(question: str, results: list[Any], resume_text: str) -> str | None:
    api_key = get_google_api_key()
    if not api_key:
        return None
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        return None

    result_lines = []
    for result in results[:10]:
        result_lines.append(
            f"Rank {result.rank}: {result.filename}; final={result.final_score:.2f}%; "
            f"semantic={result.semantic_score:.2f}%; label={result.recommendation}; "
            f"matched_resume_skills={', '.join(result.matched_skills[:12]) or 'none'}; "
            f"missing_job_skills={', '.join(result.missing_skills[:12]) or 'none'}; "
            f"experience={result.detected_resume_years:g}/{result.required_years:g} years."
        )
    prompt = PromptTemplate.from_template(JOB_MATCH_CHATBOT_TEMPLATE)
    model_name = resolve_gemini_chat_model(api_key, os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_CHAT_MODEL))
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
    )
    chain = prompt | llm
    try:
        response = chain.invoke(
            {
                "resume_text": shorten_text(resume_text, 4000),
                "results_summary": "\n".join(result_lines),
                "question": question,
            }
        )
        return getattr(response, "content", str(response)).strip()
    except Exception:
        return None
