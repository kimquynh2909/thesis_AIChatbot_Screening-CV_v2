from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from config.settings import DEFAULT_EMBEDDING_MODELS
from src.models.embedding_matcher import cosine
from src.preprocessing.chunker import aggregate_chunk_scores, chunk_text
from src.preprocessing.text_cleaner import clean_for_matching


@dataclass
class GoogleEmbeddingMatcher:
    model_name: str = DEFAULT_EMBEDDING_MODELS["google"]
    name: str = "Google Embeddings"
    preferred_fallback_models: tuple[str, ...] = (
        "models/gemini-embedding-001",
        "models/gemini-embedding-2",
        "models/gemini-embedding-2-preview",
    )

    def __post_init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        if not self.api_key:
            raise RuntimeError("Google Embeddings require GOOGLE_API_KEY or GEMINI_API_KEY in .env.")
        if not resume_texts:
            return []
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("Install google-generativeai to use Google Embeddings.") from exc

        genai.configure(api_key=self.api_key)
        self.model_name = self._resolve_model_name(genai)
        jd_embedding = self._embed(genai, clean_for_matching(jd_text), task_type="retrieval_query")
        scores: list[float] = []
        for resume_text in resume_texts:
            chunks = chunk_text(clean_for_matching(resume_text))
            if not chunks:
                scores.append(0.0)
                continue
            chunk_scores = []
            for chunk in chunks:
                embedding = self._embed(genai, chunk.text, task_type="retrieval_document")
                similarity = cosine(jd_embedding, embedding)
                chunk_scores.append(float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)))
            scores.append(aggregate_chunk_scores(chunk_scores))
        return scores

    def _embed(self, genai, text: str, task_type: str) -> np.ndarray:
        response = genai.embed_content(model=self.model_name, content=text, task_type=task_type)
        vector = response.get("embedding")
        if vector is None:
            raise RuntimeError("Google embedding API did not return an embedding vector.")
        return np.asarray(vector, dtype=float)

    def _resolve_model_name(self, genai) -> str:
        """Use the configured model if available, otherwise choose an accessible embedding model."""
        try:
            available = []
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "embedContent" in methods:
                    available.append(model.name)
        except Exception:
            return self.model_name

        if self.model_name in available:
            return self.model_name
        for model_name in self.preferred_fallback_models:
            if model_name in available:
                return model_name
        if available:
            return available[0]
        raise RuntimeError("No Gemini embedding model with embedContent support is available for this API key.")
