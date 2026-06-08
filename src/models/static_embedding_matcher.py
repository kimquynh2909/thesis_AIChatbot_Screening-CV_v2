from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.models.embedding_matcher import cosine
from src.preprocessing.text_cleaner import tokenize_words

VectorMapping = Mapping[str, Sequence[float] | np.ndarray]


@lru_cache(maxsize=4)
def load_static_vectors(model_id: str, vector_path: str | None = None):
    """Load pretrained static word vectors from a local file or gensim-data id."""
    try:
        from gensim.models import KeyedVectors
    except ImportError as exc:
        raise ImportError(
            "gensim is required for Word2Vec and GloVe baselines. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if vector_path:
        path = Path(vector_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Static embedding file not found: {path}")
        return _load_keyed_vectors(KeyedVectors, path)

    try:
        import gensim.downloader as api
    except ImportError as exc:
        raise ImportError("gensim downloader is required to load pretrained static embeddings.") from exc

    return api.load(model_id)


def _load_keyed_vectors(keyed_vectors_cls, path: Path):
    suffix = path.suffix.lower()
    if suffix in {".kv", ".model"}:
        return keyed_vectors_cls.load(str(path), mmap="r")

    try:
        return keyed_vectors_cls.load(str(path), mmap="r")
    except Exception:
        binary = suffix == ".bin"

    try:
        return keyed_vectors_cls.load_word2vec_format(str(path), binary=binary)
    except ValueError:
        if binary:
            raise
        return keyed_vectors_cls.load_word2vec_format(str(path), binary=False, no_header=True)


@dataclass
class StaticEmbeddingMatcher:
    """Mean-pooled static word embedding baseline."""

    model_id: str
    name: str
    vector_path: str | None = None
    vectors: VectorMapping | None = field(default=None, repr=False, compare=False)

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []

        vectors = self.vectors if self.vectors is not None else load_static_vectors(self.model_id, self.vector_path)
        jd_embedding = self._document_embedding(jd_text, vectors)
        if jd_embedding is None:
            return [0.0 for _ in resume_texts]

        scores: list[float] = []
        for resume_text in resume_texts:
            resume_embedding = self._document_embedding(resume_text, vectors)
            if resume_embedding is None:
                scores.append(0.0)
                continue
            scores.append(self._cosine_01(jd_embedding, resume_embedding))
        return scores

    def _document_embedding(self, text: str, vectors) -> np.ndarray | None:
        token_vectors: list[np.ndarray] = []
        expected_dim: int | None = None
        for token in tokenize_words(text):
            vector = self._lookup_vector(token, vectors)
            if vector is None:
                continue
            if expected_dim is None:
                expected_dim = int(vector.shape[0])
            if vector.shape[0] != expected_dim:
                continue
            token_vectors.append(vector)

        if not token_vectors:
            return None
        return np.mean(np.vstack(token_vectors), axis=0)

    @staticmethod
    def _lookup_vector(token: str, vectors) -> np.ndarray | None:
        for candidate in _token_candidates(token):
            if candidate in vectors:
                return np.asarray(vectors[candidate], dtype=float)
        return None

    @staticmethod
    def _cosine_01(a: np.ndarray, b: np.ndarray) -> float:
        value = cosine(a, b)
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def _token_candidates(token: str) -> list[str]:
    candidates = [token]
    lower = token.lower()
    title = token.title()
    upper = token.upper()
    for value in [lower, title, upper]:
        if value not in candidates:
            candidates.append(value)
    return candidates
