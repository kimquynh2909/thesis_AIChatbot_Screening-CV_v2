from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config.settings import (
    QDRANT_API_KEY,
    QDRANT_JOB_COLLECTION_NAME,
    QDRANT_JOB_EMBEDDING_MODEL,
    QDRANT_URL,
)
from src.llm.structured_extraction import extract_cv_json, extract_job_description_json
from src.models.embedding_matcher import load_sentence_transformer
from src.preprocessing.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import clean_for_matching
from src.utils.file_utils import shorten_text


@dataclass(frozen=True)
class QdrantJobRAGResult:
    rank: int
    score: float
    chunk_index: int
    chunk_text: str
    job_id: str | None = None
    chunk_id: str | None = None
    job_title: str | None = None
    category: str | None = None
    job_category: str | None = None
    required_skills: str | None = None
    job_description: str | None = None
    text: str | None = None
    candidate_name: str | None = None
    resume_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantJobRetriever:
    """
    Retrieve full job-description metadata from Qdrant.

    Main SBERT explainability flow:
        clean full resume text -> LLM/deterministic CV field extraction ->
        one SBERT query vector -> Qdrant top-k jobs -> full payload metadata.

    The query is not chunked. Retrieved Qdrant metadata is used only as
    explainability/debugging context and does not change SBERT ranking scores.
    """

    def __init__(
        self,
        collection_name: str = QDRANT_JOB_COLLECTION_NAME,
        model_name: str = QDRANT_JOB_EMBEDDING_MODEL,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
    ) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "qdrant-client is required for Qdrant job retrieval. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

            if self.qdrant_url:
                self._client = QdrantClient(url=self.qdrant_url, api_key=self.api_key or None, timeout=120)
            else:
                raise ValueError("QDRANT_URL is required for the job-description Qdrant collection.")
        return self._client

    def collection_count(self) -> int:
        return int(self.client.count(collection_name=self.collection_name).count)

    def retrieve_job_description_evidence(
        self,
        job_description_text: str,
        top_k: int = 5,
        candidate_name: str | None = None,
        resume_id: str | None = None,
        job_extraction: dict[str, Any] | None = None,
        use_llm_extraction: bool = True,
        skill_extractor: SkillExtractor | None = None,
    ) -> list[QdrantJobRAGResult]:
        cleaned_text = clean_for_matching(job_description_text)
        extraction = job_extraction or extract_job_description_json(
            cleaned_text,
            use_llm=use_llm_extraction,
            skill_extractor=skill_extractor,
        )
        query_text = self._build_job_query_text(cleaned_text, extraction)
        results = self._search(query_text=query_text, top_k=top_k, candidate_name=candidate_name, resume_id=resume_id)
        self._print_debug(candidate_name or "job_description", top_k, cleaned_text, extraction, results, query_label="Job description")
        return results

    def retrieve_resume_job_evidence(
        self,
        resume_text: str,
        top_k: int = 5,
        candidate_name: str | None = None,
        resume_id: str | None = None,
        resume_extraction: dict[str, Any] | None = None,
        use_llm_extraction: bool = True,
        skill_extractor: SkillExtractor | None = None,
    ) -> list[QdrantJobRAGResult]:
        """Retrieve top-K Qdrant jobs using one cleaned full-resume SBERT query.

        SBERT still calculates the main semantic ranking score elsewhere. These
        Qdrant results are only evidence for explainability/debugging and never
        modify the final ranking score.
        """
        cleaned_text = clean_for_matching(resume_text)
        extraction = resume_extraction or extract_cv_json(
            cleaned_text,
            jd_extraction={},
            use_llm=use_llm_extraction,
            skill_extractor=skill_extractor,
        )
        query_text = self._build_resume_query_text(cleaned_text, extraction)
        results = self._search(query_text=query_text, top_k=top_k, candidate_name=candidate_name, resume_id=resume_id)
        self._print_debug(candidate_name or "resume", top_k, cleaned_text, extraction, results, query_label="Resume")
        return results

    def _search(
        self,
        query_text: str,
        top_k: int,
        candidate_name: str | None,
        resume_id: str | None,
    ) -> list[QdrantJobRAGResult]:
        top_k = max(1, min(int(top_k or 5), 10))
        query_vector = self._embed_query(query_text).astype(float).tolist()

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        except AttributeError:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        return [
            self._point_to_result(point, rank=index, candidate_name=candidate_name, resume_id=resume_id)
            for index, point in enumerate(points, start=1)
        ]

    def _embed_query(self, query_text: str) -> np.ndarray:
        model = load_sentence_transformer(self.model_name)
        vector = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=float)

    def _build_job_query_text(self, cleaned_text: str, extraction: dict[str, Any]) -> str:
        sections = [
            self._field_to_text("required skills", extraction.get("required_skills", [])),
            self._field_to_text("preferred skills", extraction.get("preferred_skills", [])),
            self._field_to_text("tools", extraction.get("tools", [])),
            self._field_to_text("education", extraction.get("education", [])),
            self._field_to_text("experience", extraction.get("experience", [])),
            f"job description: {cleaned_text}",
        ]
        return "\n".join(section for section in sections if section.strip())

    def _build_resume_query_text(self, cleaned_text: str, extraction: dict[str, Any]) -> str:
        sections = [
            self._field_to_text("resume skills", extraction.get("skills", [])),
            self._field_to_text("education", extraction.get("education", [])),
            self._field_to_text("experience", extraction.get("experience", [])),
            self._field_to_text("projects", extraction.get("projects", [])),
            f"resume full text: {cleaned_text}",
        ]
        return "\n".join(section for section in sections if section.strip())

    def _point_to_result(
        self,
        point: Any,
        rank: int,
        candidate_name: str | None,
        resume_id: str | None,
    ) -> QdrantJobRAGResult:
        payload = dict(getattr(point, "payload", None) or {})
        chunk_text = str(payload.get("text") or payload.get("job_description") or "")
        category = _optional_str(payload.get("category"))
        return QdrantJobRAGResult(
            rank=rank,
            score=round(float(getattr(point, "score", 0.0) or 0.0), 4),
            chunk_index=_safe_int(payload.get("chunk_index")),
            chunk_text=chunk_text,
            job_id=_optional_str(payload.get("job_id")),
            chunk_id=_optional_str(payload.get("chunk_id")),
            job_title=_optional_str(payload.get("job_title")),
            category=category,
            job_category=category,
            required_skills=_optional_str(payload.get("required_skills")),
            job_description=_optional_str(payload.get("job_description")),
            text=_optional_str(payload.get("text")),
            candidate_name=candidate_name,
            resume_id=resume_id,
            payload=payload,
        )

    @staticmethod
    def _field_to_text(label: str, value: Any) -> str:
        items = _string_list(value)
        return f"{label}: {', '.join(items)}" if items else ""

    def _print_debug(
        self,
        candidate: str,
        top_k: int,
        cleaned_text: str,
        extraction: dict[str, Any],
        results: list[QdrantJobRAGResult],
        query_label: str = "Resume",
    ) -> None:
        print("\n================= QDRANT JOB RAG DEBUG =================", flush=True)
        print(f"Collection: {self.collection_name}", flush=True)
        print(f"Embedding model: {self.model_name}", flush=True)
        print(f"Query source: {query_label}", flush=True)
        print(f"Candidate: {candidate}", flush=True)
        print(f"Top K: {top_k}", flush=True)
        print(f"Cleaned {query_label} Preview: {shorten_text(cleaned_text, 160)}", flush=True)
        print(f"Extracted fields: {extraction}", flush=True)

        for result in results:
            print("", flush=True)
            print(
                f"[QDRANT-{result.rank}] score={result.score:.4f} "
                f"job_id={result.job_id} title={result.job_title} category={result.category}",
                flush=True,
            )
            print(shorten_text(result.chunk_text, 280), flush=True)

        print("\n========================================================\n", flush=True)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("name") or item.get("requirement") or item.get("text") or item.get("value")
            else:
                text = item
            if text:
                result.append(str(text).strip())
        return [item for item in result if item]
    text = str(value).strip()
    return [text] if text else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def check_qdrant_resume_retrieval(
    query_text: str,
    top_k: int = 3,
    use_llm_extraction: bool = True,
) -> list[QdrantJobRAGResult]:
    """Small helper for terminal checks without printing secrets."""
    retriever = QdrantJobRetriever()
    try:
        count = retriever.collection_count()
        print(f"Qdrant collection '{retriever.collection_name}' count: {count}", flush=True)
    except Exception as exc:
        print(f"Could not count Qdrant collection: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    return retriever.retrieve_resume_job_evidence(query_text, top_k=top_k, use_llm_extraction=use_llm_extraction)


def check_qdrant_job_retrieval(
    query_text: str,
    top_k: int = 3,
    use_llm_extraction: bool = True,
    query_type: str = "resume",
) -> list[QdrantJobRAGResult]:
    """Compatibility helper for terminal checks without printing secrets."""
    retriever = QdrantJobRetriever()
    try:
        count = retriever.collection_count()
        print(f"Qdrant collection '{retriever.collection_name}' count: {count}", flush=True)
    except Exception as exc:
        print(f"Could not count Qdrant collection: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    if query_type.lower() in {"jd", "job", "job_description", "job-description"}:
        return retriever.retrieve_job_description_evidence(
            query_text,
            top_k=top_k,
            use_llm_extraction=use_llm_extraction,
        )
    return retriever.retrieve_resume_job_evidence(query_text, top_k=top_k, use_llm_extraction=use_llm_extraction)
