from __future__ import annotations

from dataclasses import dataclass

from src.services.vector_store_service import (
    VectorSearchResult,
    build_rag_context,
    build_rag_documents,
    build_vector_chunks,
)


@dataclass
class ResumeDocument:
    candidate_id: str
    filename: str
    text: str


def test_build_rag_documents_includes_jd_and_resumes() -> None:
    documents = build_rag_documents(
        "Need Python and NLP.",
        [ResumeDocument("candidate_1", "alice.txt", "Python engineer with NLP experience.")],
    )
    assert [document.document_type for document in documents] == ["job_description", "resume"]
    assert documents[1].metadata["candidate_id"] == "candidate_1"


def test_build_vector_chunks_creates_stable_payload() -> None:
    documents = build_rag_documents(
        "Need Python and NLP.",
        [ResumeDocument("candidate_1", "alice.txt", "Python engineer with NLP experience.")],
    )
    chunks = build_vector_chunks(documents, "test_collection", max_words=4, overlap_words=1)
    assert len(chunks) >= 2
    assert chunks[0].point_id == build_vector_chunks(documents, "test_collection", max_words=4, overlap_words=1)[0].point_id
    payload = chunks[-1].payload()
    assert payload["document_type"] == "resume"
    assert payload["filename"] == "alice.txt"
    assert "text" in payload


def test_build_rag_context_formats_sources() -> None:
    context = build_rag_context(
        [
            VectorSearchResult(
                score=0.91,
                text="Python NLP project evidence.",
                document_id="candidate_1",
                document_type="resume",
                filename="alice.txt",
                chunk_id=0,
                metadata={},
            )
        ]
    )
    assert "alice.txt" in context
    assert "Python NLP project evidence." in context
