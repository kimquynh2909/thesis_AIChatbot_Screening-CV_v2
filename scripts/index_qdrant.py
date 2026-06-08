from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import QDRANT_COLLECTION_NAME, SAMPLE_JD_DIR, SAMPLE_RESUME_DIR
from src.preprocessing.document_parser import DocumentParser
from src.services.matching_service import ResumeDocument
from src.services.vector_store_service import QdrantRAGStore, build_rag_documents


def load_sample_documents() -> tuple[str, list[ResumeDocument]]:
    parser = DocumentParser()
    jd_files = sorted(SAMPLE_JD_DIR.glob("*.txt"))
    resume_files = sorted(SAMPLE_RESUME_DIR.glob("*.txt"))
    if not jd_files or not resume_files:
        raise FileNotFoundError("Sample JD/resume files are missing.")
    jd_text = parser.parse_path(jd_files[0]).cleaned_text
    resumes = [
        ResumeDocument(candidate_id=path.stem, filename=path.name, text=parser.parse_path(path).cleaned_text)
        for path in resume_files
    ]
    return jd_text, resumes


def load_user_documents(jd_path: Path, resume_paths: list[Path]) -> tuple[str, list[ResumeDocument]]:
    parser = DocumentParser()
    jd_text = parser.parse_path(jd_path).cleaned_text
    resumes = [
        ResumeDocument(candidate_id=path.stem, filename=path.name, text=parser.parse_path(path).cleaned_text)
        for path in resume_paths
    ]
    return jd_text, resumes


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Index resume/JD chunks into Qdrant for RAG experiments.")
    arg_parser.add_argument("--jd", type=Path, help="Path to a job description file.")
    arg_parser.add_argument("--resumes", type=Path, nargs="*", default=[], help="Resume files to index.")
    arg_parser.add_argument("--collection", default=QDRANT_COLLECTION_NAME, help="Qdrant collection name.")
    args = arg_parser.parse_args()

    if args.jd and args.resumes:
        jd_text, resumes = load_user_documents(args.jd, args.resumes)
    else:
        jd_text, resumes = load_sample_documents()

    store = QdrantRAGStore(collection_name=args.collection)
    count = store.index_documents(build_rag_documents(jd_text, resumes))
    print(f"Indexed {count} chunks into Qdrant collection '{args.collection}'.")


if __name__ == "__main__":
    main()
