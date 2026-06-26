from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.qdrant_job_retriever import check_qdrant_resume_retrieval
from src.utils.file_utils import read_text_file, shorten_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Qdrant resume-to-job retrieval and print returned metadata.")
    parser.add_argument("--query", default="", help="Resume/CV text to search with.")
    parser.add_argument("--file", type=Path, help="Path to a resume/CV text file.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Qdrant job results to retrieve.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic field extraction for the check.")
    args = parser.parse_args()

    if args.file:
        query_text = read_text_file(args.file)
    elif args.query:
        query_text = args.query
    else:
        query_text = (
            "Human resources generalist with experience in employee relations, talent acquisition, "
            "performance management, onboarding, HR compliance, policy administration, and HRIS reporting."
        )

    results = check_qdrant_resume_retrieval(query_text=query_text, top_k=args.top_k, use_llm_extraction=not args.no_llm)
    if not results:
        print("No Qdrant results returned.")
        return

    print("\nTop Qdrant results:")
    for result in results:
        print("-" * 80)
        print(f"Rank: {result.rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Job ID: {result.job_id}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Chunk Index: {result.chunk_index}")
        print(f"Job Title: {result.job_title}")
        print(f"Category: {result.category}")
        print(f"Required Skills: {shorten_text(result.required_skills or '', 220)}")
        print(f"Job Description: {shorten_text(result.job_description or '', 420)}")
        print(f"Text: {shorten_text(result.text or '', 420)}")


if __name__ == "__main__":
    main()
