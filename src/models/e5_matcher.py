from __future__ import annotations

from dataclasses import dataclass

from config.settings import DEFAULT_EMBEDDING_MODELS
from src.models.embedding_matcher import SentenceEmbeddingMatcher


@dataclass
class E5Matcher(SentenceEmbeddingMatcher):
    model_name: str = DEFAULT_EMBEDDING_MODELS["e5"]
    name: str = "E5"
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
