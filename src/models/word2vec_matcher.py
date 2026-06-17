from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import DEFAULT_EMBEDDING_MODELS, STATIC_EMBEDDING_PATHS
from src.preprocessing.text_cleaner import clean_for_matching

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]

word2vec_path = PROJECT_ROOT / "data" / "embeddings" / "GoogleNews-vectors-negative300.bin"


# @lru_cache(maxsize=2)
# def load_word2vec_vectors(
#     model_id: str,
#     vector_path: str | None = None,
# ) -> Any:
#     """
#     Load Word2Vec vectors from a local file or from gensim-data.

#     Priority:
#     1. If vector_path is provided and exists, load local vectors.
#     2. Otherwise, load the pretrained model from gensim downloader.

#     The result is cached so the model is loaded only once.
#     """
#     try:
#         from gensim.models import KeyedVectors
#     except ImportError as exc:
#         raise ImportError(
#             "gensim is required for the Word2Vec baseline. "
#             "Install it with: pip install gensim"
#         ) from exc

#     if vector_path:
#         path = Path(vector_path).expanduser()

#         if not path.exists():
#             raise FileNotFoundError(f"Word2Vec vector file not found: {path}")

#         print(f"Loading Word2Vec vectors from local file: {path}")

#         suffix = path.suffix.lower()

#         if suffix in {".kv", ".model"}:
#             return KeyedVectors.load(str(path), mmap="r")

#         try:
#             return KeyedVectors.load(str(path), mmap="r")
#         except Exception:
#             pass

#         binary = suffix in {".bin", ".gz"}

#         try:
#             return KeyedVectors.load_word2vec_format(
#                 str(path),
#                 binary=binary,
#             )
#         except ValueError:
#             if binary:
#                 raise

#             return KeyedVectors.load_word2vec_format(
#                 str(path),
#                 binary=False,
#                 no_header=True,
#             )

#     try:
#         import gensim.downloader as api
#     except ImportError as exc:
#         raise ImportError(
#             "gensim downloader is required to load pretrained Word2Vec models."
#         ) from exc

#     print(f"Loading Word2Vec model from gensim: {model_id}")
#     return api.load(model_id)

@lru_cache(maxsize=1)
def load_local_word2vec_vectors() -> Any:
    if not word2vec_path.exists():
        raise FileNotFoundError(f"Word2Vec file not found: {word2vec_path}")

    print(f"Loading Word2Vec vectors from: {word2vec_path}")

    return KeyedVectors.load_word2vec_format(
        str(word2vec_path),
        binary=True,
    )

@dataclass
class Word2VecMatcher:
    """
    Word2Vec baseline matcher for resume-job description matching.

    Algorithm:
    1. Clean job description and resume text.
    2. Tokenize text into words.
    3. Convert each token into a pretrained Word2Vec vector.
    4. Average all valid token vectors into one document vector.
    5. Compute cosine similarity between JD vector and resume vector.

    Output:
    - A score in the range [0, 1].
    - Higher score means higher CV-JD similarity.
    """

    model_id: str = DEFAULT_EMBEDDING_MODELS.get(
        "word2vec",
        "word2vec-google-news-300",
    )

    name: str = "Word2Vec Average Embeddings"

    vector_path: str | None = STATIC_EMBEDDING_PATHS.get("word2vec")

    min_token_length: int = 2

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        """
        Compute matching scores between one job description and many resumes.

        Args:
            jd_text: Job description text.
            resume_texts: List of resume texts.

        Returns:
            List of similarity scores. Each score is in [0, 1].
        """
        if not resume_texts:
            return []

        vectors = load_local_word2vec_vectors()

        jd_vector = self._document_vector(jd_text, vectors)

        if jd_vector is None:
            return [0.0 for _ in resume_texts]

        scores: list[float] = []

        for resume_text in resume_texts:
            resume_vector = self._document_vector(resume_text, vectors)

            if resume_vector is None:
                scores.append(0.0)
                continue

            score = cosine_similarity(
                jd_vector.reshape(1, -1),
                resume_vector.reshape(1, -1),
            )[0][0]
            #score = self._cosine_similarity_01(jd_vector, resume_vector)
            scores.append(score)

        return scores

    def _document_vector(self, text: str, vectors: Any) -> np.ndarray | None:
        """
        Convert a document into a single vector by mean-pooling Word2Vec token vectors.

        If no valid token vectors are found, return None.
        """
        tokens = self._tokenize(text)

        token_vectors: list[np.ndarray] = []

        for token in tokens:
            vector = self._lookup_vector(token, vectors)

            if vector is None:
                continue

            token_vectors.append(vector)

        if not token_vectors:
            return None

        document_vector = np.mean(
            np.vstack(token_vectors),
            axis=0,
        ).astype(np.float32)

        if np.linalg.norm(document_vector) == 0:
            return None

        return document_vector

    def _tokenize(self, text: str) -> list[str]:
        """
        Clean text and using a regex, extract candidate tokens that are likely to be in the Word2Vec vocabulary.
        """
        cleaned_text = clean_for_matching(str(text))

        tokens = re.findall(
            r"[a-zA-Z][a-zA-Z0-9+#.\-]*",
            cleaned_text,
        )

        return [
            token
            for token in tokens
            if len(token.strip()) >= self.min_token_length
        ]

    def _lookup_vector(self, token: str, vectors: Any) -> np.ndarray | None:
        """
        Look up a token vector using multiple casing candidates.

        This is useful because pretrained Word2Vec may contain:
        - Python instead of python
        - SQL instead of sql
        - Java instead of java
        """
        for candidate in self._token_candidates(token):
            if candidate in vectors:
                return np.asarray(vectors[candidate], dtype=np.float32)

        return None

    @staticmethod
    def _token_candidates(token: str) -> list[str]:
        """
        Generate possible token forms for pretrained vector lookup.
        """
        token = str(token).strip()

        candidates = [
            token,
            token.lower(),
            token.title(),
            token.upper(),
        ]

        # Extra normalized variants for common technical tokens.
        normalized_variants = {
            "c++": "cpp",
            "c#": "csharp",
            ".net": "dotnet",
            "node.js": "nodejs",
            "react.js": "reactjs",
            "vue.js": "vuejs",
            "next.js": "nextjs",
        }

        lower = token.lower()

        if lower in normalized_variants:
            candidates.append(normalized_variants[lower])

        # Remove duplicates while preserving order.
        unique_candidates: list[str] = []

        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)

        return unique_candidates

    @staticmethod
    def _cosine_similarity_01(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity and clip it to [0, 1].

        Important:
        Do not use (cosine + 1) / 2 here because it artificially inflates
        similarity scores. For example, raw cosine 0.4 would become 0.7,
        which may cause almost all CV-JD pairs to be predicted as positive.
        """
        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        cosine_value = float(np.dot(a, b) / denominator)

        return float(np.clip(cosine_value, 0.0, 1.0))