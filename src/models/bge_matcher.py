from __future__ import annotations

from dataclasses import dataclass

from config.settings import DEFAULT_EMBEDDING_MODELS
from src.models.embedding_matcher import SentenceEmbeddingMatcher


@dataclass
class BGEMatcher(SentenceEmbeddingMatcher):
    model_name: str = DEFAULT_EMBEDDING_MODELS["bge"]
    name: str = "BGE"
