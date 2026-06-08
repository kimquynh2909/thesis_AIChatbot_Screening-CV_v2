from __future__ import annotations

from dataclasses import dataclass

from config.settings import DEFAULT_EMBEDDING_MODELS, STATIC_EMBEDDING_PATHS
from src.models.static_embedding_matcher import StaticEmbeddingMatcher


@dataclass
class Word2VecMatcher(StaticEmbeddingMatcher):
    model_id: str = DEFAULT_EMBEDDING_MODELS["word2vec"]
    name: str = "Word2Vec Average Embeddings"
    vector_path: str | None = STATIC_EMBEDDING_PATHS["word2vec"]
