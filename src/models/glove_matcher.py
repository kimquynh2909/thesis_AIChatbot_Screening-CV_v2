from __future__ import annotations

from dataclasses import dataclass

from config.settings import DEFAULT_EMBEDDING_MODELS, STATIC_EMBEDDING_PATHS
from src.models.static_embedding_matcher import StaticEmbeddingMatcher


@dataclass
class GloveMatcher(StaticEmbeddingMatcher):
    model_id: str = DEFAULT_EMBEDDING_MODELS["glove"]
    name: str = "GloVe Average Embeddings"
    vector_path: str | None = STATIC_EMBEDDING_PATHS["glove"]
