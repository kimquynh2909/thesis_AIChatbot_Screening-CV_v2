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


def answer_hr_question(question: str, results: list[Any], jd_text: str, use_llm: bool = True) -> str:
    if not question.strip():
        return "Please enter a question about the screening results."

    if use_llm:
        answer = _try_llm_answer(question, results, jd_text)
        if answer:
            return answer
    return deterministic_answer(question, results)


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
