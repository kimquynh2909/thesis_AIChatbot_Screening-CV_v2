from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from config.settings import (
    DEFAULT_RAG_EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
)
from src.models.embedding_matcher import load_sentence_transformer
from src.preprocessing.chunker import chunk_text
from src.preprocessing.text_cleaner import clean_for_matching
from src.utils.file_utils import shorten_text

if TYPE_CHECKING:
    from src.services.matching_service import JobDescriptionDocument, ResumeDocument


@dataclass(frozen=True)
class RAGDocument:
    document_id: str
    document_type: str
    filename: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorChunk:
    point_id: str
    document_id: str
    document_type: str
    filename: str
    chunk_id: int
    text: str
    metadata: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "text": self.text,
            **self.metadata,
        }


@dataclass(frozen=True)
class VectorSearchResult:
    score: float
    text: str
    document_id: str
    document_type: str
    filename: str
    chunk_id: int
    metadata: dict[str, Any]


class LocalEmbeddingEncoder:
    """SentenceTransformer encoder used for Qdrant indexing and retrieval."""

    def __init__(self, model_name: str = DEFAULT_RAG_EMBEDDING_MODEL):
        self.model_name = model_name

    @property
    def vector_size(self) -> int:
        sample = self.embed_documents(["dimension probe"])
        return int(sample.shape[1])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, task="document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], task="query")[0]

    def _encode(self, texts: list[str], task: str) -> np.ndarray:
        model = load_sentence_transformer(self.model_name)
        prepared = [self._prepare_text(text, task) for text in texts]
        vectors = model.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)

    def _prepare_text(self, text: str, task: str) -> str:
        cleaned = clean_for_matching(text)
        if "e5" not in self.model_name.lower():
            return cleaned
        prefix = "query: " if task == "query" else "passage: "
        return prefix + cleaned


class QdrantRAGStore:
    """Optional Qdrant vector store for RAG-ready resume/JD chunks."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION_NAME,
        encoder: LocalEmbeddingEncoder | None = None,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
    ):
        self.collection_name = collection_name
        self.encoder = encoder or LocalEmbeddingEncoder()
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            QdrantClient = import_qdrant_client()

            if self.qdrant_url:
                self._client = QdrantClient(url=self.qdrant_url, api_key=self.api_key or None)
            else:
                QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(QDRANT_LOCAL_PATH))
        return self._client

    def ensure_collection(self) -> None:
        models = import_qdrant_models()

        vector_size = self.encoder.vector_size
        if self._collection_exists():
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def index_documents(self, documents: list[RAGDocument], max_words: int = 220, overlap_words: int = 40) -> int:
        models = import_qdrant_models()

        chunks = build_vector_chunks(documents, self.collection_name, max_words=max_words, overlap_words=overlap_words)
        if not chunks:
            return 0

        self.ensure_collection()
        vectors = self.encoder.embed_documents([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector=vectors[idx].astype(float).tolist(),
                payload=chunk.payload(),
            )
            for idx, chunk in enumerate(chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query: str, limit: int = 5, document_type: str | None = None) -> list[VectorSearchResult]:
        self.ensure_collection()
        query_vector = self.encoder.embed_query(query).astype(float).tolist()
        query_filter = self._document_type_filter(document_type)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except AttributeError:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )

        return [point_to_search_result(point) for point in points]

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            try:
                self.client.get_collection(self.collection_name)
                return True
            except Exception:
                return False
        except Exception:
            return False

    @staticmethod
    def _document_type_filter(document_type: str | None):
        if not document_type:
            return None
        models = import_qdrant_models()

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(value=document_type),
                )
            ]
        )


def import_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for Qdrant RAG indexing. "
            "Install it in the active environment with: python -m pip install qdrant-client"
        ) from exc
    return QdrantClient


def import_qdrant_models():
    try:
        from qdrant_client.http import models
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for Qdrant RAG indexing. "
            "Install it in the active environment with: python -m pip install qdrant-client"
        ) from exc
    return models


def build_rag_documents(jd_text: str, resumes: list[ResumeDocument]) -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    if jd_text and jd_text.strip():
        documents.append(
            RAGDocument(
                document_id="job_description",
                document_type="job_description",
                filename="job_description.txt",
                text=jd_text,
            )
        )

    for resume in resumes:
        documents.append(
            RAGDocument(
                document_id=resume.candidate_id,
                document_type="resume",
                filename=resume.filename,
                text=resume.text,
                metadata={"candidate_id": resume.candidate_id},
            )
        )
    return documents


def build_job_matching_rag_documents(
    resume: ResumeDocument,
    job_descriptions: list[JobDescriptionDocument],
) -> list[RAGDocument]:
    documents: list[RAGDocument] = []
    if resume.text and resume.text.strip():
        documents.append(
            RAGDocument(
                document_id=resume.candidate_id,
                document_type="resume",
                filename=resume.filename,
                text=resume.text,
                metadata={"candidate_id": resume.candidate_id},
            )
        )

    for job in job_descriptions:
        documents.append(
            RAGDocument(
                document_id=job.job_id,
                document_type="job_description",
                filename=job.filename,
                text=job.text,
                metadata={"job_id": job.job_id},
            )
        )
    return documents


def build_vector_chunks(
    documents: list[RAGDocument],
    collection_name: str,
    max_words: int = 220,
    overlap_words: int = 40,
) -> list[VectorChunk]:
    chunks: list[VectorChunk] = []
    for document in documents:
        for chunk in chunk_text(document.text, max_words=max_words, overlap_words=overlap_words):
            point_id = stable_point_id(collection_name, document.document_id, chunk.chunk_id)
            chunks.append(
                VectorChunk(
                    point_id=point_id,
                    document_id=document.document_id,
                    document_type=document.document_type,
                    filename=document.filename,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=document.metadata,
                )
            )
    return chunks


def stable_point_id(collection_name: str, document_id: str, chunk_id: int) -> str:
    value = f"{collection_name}:{document_id}:{chunk_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def point_to_search_result(point: Any) -> VectorSearchResult:
    payload = getattr(point, "payload", None) or {}
    base_keys = {"document_id", "document_type", "filename", "chunk_id", "text"}
    metadata = {key: value for key, value in payload.items() if key not in base_keys}
    return VectorSearchResult(
        score=float(getattr(point, "score", 0.0) or 0.0),
        text=str(payload.get("text", "")),
        document_id=str(payload.get("document_id", "")),
        document_type=str(payload.get("document_type", "")),
        filename=str(payload.get("filename", "")),
        chunk_id=int(payload.get("chunk_id", 0) or 0),
        metadata=metadata,
    )


def build_rag_context(results: list[VectorSearchResult], max_chars: int = 3500) -> str:
    sections: list[str] = []
    used = 0
    for result in results:
        header = f"[{result.document_type} | {result.filename} | chunk {result.chunk_id} | score {result.score:.3f}]"
        body = shorten_text(result.text, max_chars=max(200, max_chars - used - len(header) - 2))
        section = f"{header}\n{body}"
        if used + len(section) > max_chars and sections:
            break
        sections.append(section)
        used += len(section)
        if used >= max_chars:
            break
    return "\n\n".join(sections)
