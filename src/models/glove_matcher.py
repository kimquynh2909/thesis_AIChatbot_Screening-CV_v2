from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import KeyedVectors

from config.settings import DEFAULT_EMBEDDING_MODELS, STATIC_EMBEDDING_PATHS
from src.preprocessing.text_cleaner import clean_for_matching


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GLOVE_PATH = (
    PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"
)


ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
    "of", "on", "or", "our", "she", "that", "the", "their", "them",
    "they", "this", "to", "was", "we", "will", "with", "you", "your",
    "about", "above", "after", "again", "against", "all", "am", "any",
    "because", "been", "before", "being", "below", "between", "both",
    "but", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "further", "had", "having", "here", "how", "into",
    "more", "most", "no", "nor", "not", "now", "off", "once", "only",
    "other", "out", "over", "own", "same", "should", "so", "some",
    "such", "than", "then", "there", "these", "those", "through",
    "too", "under", "until", "up", "very", "what", "when", "where",
    "which", "while", "who", "whom", "why",
}


GENERIC_RECRUITMENT_WORDS = {
    "job", "role", "position", "candidate", "company", "organization",
    "business", "client", "customer", "team", "teams", "department",
    "responsible", "responsibility", "responsibilities", "require",
    "requires", "required", "requirement", "requirements", "ability",
    "abilities", "skill", "skills", "strong", "excellent", "good",
    "great", "work", "working", "worked", "experience", "experienced",
    "knowledge", "understanding", "passion", "provide", "provides",
    "providing", "support", "supports", "supporting", "ensure",
    "ensures", "ensuring", "manage", "manages", "managing", "perform",
    "performs", "performing", "make", "making", "help", "helping",
    "include", "includes", "including", "various", "related", "field",
    "fields", "environment", "environments", "dynamic", "deliver",
    "delivering", "results", "bachelor", "bachelors", "degree",
    "certification", "certifications", "senior", "junior", "level",
}


TECH_TOKEN_VARIANTS = {
    "c++": "cpp",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "mysql": "mysql",
    "sql": "sql",
    "ml": "machine",
    "ai": "artificial",
}


@lru_cache(maxsize=1)
def load_glove_vectors(
    model_id: str = "glove-wiki-gigaword-100",
    vector_path: str | None = None,
) -> Any:
    """
    Load GloVe vectors.

    Priority:
    1. Load local glove.6B.100d.txt if the file exists.
    2. Otherwise, fallback to gensim downloader.

    Local GloVe text format:
        word value1 value2 value3 ...

    Example:
        python 0.123 -0.245 0.019 ...
    """
    path: Path | None = None

    if vector_path:
        path = Path(vector_path).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

    else:
        path = DEFAULT_GLOVE_PATH

    if path and path.exists():
        print(f"Loading GloVe vectors from local file: {path}")

        return KeyedVectors.load_word2vec_format(
            str(path),
            binary=False,
            no_header=True,
        )

    print(f"Local GloVe file not found: {path}")
    print(f"Loading GloVe model from gensim: {model_id}")

    try:
        import gensim.downloader as api
    except ImportError as exc:
        raise ImportError(
            "gensim downloader is required to load glove-wiki-gigaword-100. "
            "Install gensim or provide a local glove.6B.100d.txt file."
        ) from exc

    return api.load(model_id)


@dataclass
class GloveMatcher:
    """
    GloVe baseline matcher for CV-to-job matching.

    Correct project direction:
    - resume_text: one CV/resume
    - job_texts: many job descriptions

    Algorithm:
    1. Clean text using clean_for_matching().
    2. Tokenize text with the same style as Word2VecMatcher.
    3. Remove stopwords and generic recruitment words.
    4. Convert tokens to GloVe vectors.
    5. Build an IDF-weighted average document vector.
    6. Compute cosine similarity between the CV vector and each job vector.

    Output:
    - List of scores in range [0, 1].
    - Higher score means the job description is more relevant to the CV.
    """

    model_id: str = DEFAULT_EMBEDDING_MODELS.get(
        "glove",
        "glove-wiki-gigaword-100",
    )

    name: str = "GloVe IDF-Weighted Average Embeddings"

    vector_path: str | None = STATIC_EMBEDDING_PATHS.get("glove")

    min_token_length: int = 2

    use_idf_weighting: bool = True

    remove_stopwords: bool = True

    remove_generic_words: bool = True

    def score(self, resume_text: str, job_texts: list[str]) -> list[float]:
        """
        Compute matching scores between one resume and many job descriptions.

        Args:
            resume_text:
                One CV/resume text.
            job_texts:
                List of job description texts.

        Returns:
            List of similarity scores in [0, 1].
            The score order follows the input job_texts order.
        """
        if not job_texts:
            return []

        vectors = load_glove_vectors(
            model_id=self.model_id,
            vector_path=self.vector_path,
        )

        resume_tokens = self._tokenize(resume_text)
        job_tokens_list = [self._tokenize(job_text) for job_text in job_texts]

        if not resume_tokens or not any(job_tokens_list):
            return [0.0 for _ in job_texts]

        idf_scores = self._build_idf_scores(job_tokens_list)

        resume_vector = self._document_vector(
            tokens=resume_tokens,
            vectors=vectors,
            idf_scores=idf_scores,
        )

        if resume_vector is None:
            return [0.0 for _ in job_texts]

        scores: list[float] = []

        for job_tokens in job_tokens_list:
            job_vector = self._document_vector(
                tokens=job_tokens,
                vectors=vectors,
                idf_scores=idf_scores,
            )

            if job_vector is None:
                scores.append(0.0)
                continue

            score = self._cosine_similarity_01(resume_vector, job_vector)
            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> list[str]:
        """
        Clean text and extract useful tokens.

        This function uses the same preprocessing foundation as Word2VecMatcher:
            clean_for_matching(str(text))

        Then it removes:
        - English stopwords
        - generic recruitment words
        - very short tokens
        """
        cleaned_text = clean_for_matching(str(text)).lower()

        raw_tokens = re.findall(
            r"(?:\.net|[a-zA-Z][a-zA-Z0-9+#.\-]*)",
            cleaned_text,
        )

        tokens: list[str] = []

        for token in raw_tokens:
            normalized_token = self._normalize_token(token)

            if not normalized_token:
                continue

            if len(normalized_token.strip()) < self.min_token_length:
                continue

            if self.remove_stopwords and normalized_token in ENGLISH_STOPWORDS:
                continue

            if self.remove_generic_words and normalized_token in GENERIC_RECRUITMENT_WORDS:
                continue

            tokens.append(normalized_token)

        return tokens

    @staticmethod
    def _normalize_token(token: str) -> str:
        """
        Normalize technical tokens before vector lookup.

        Examples:
            c++      -> cpp
            c#       -> csharp
            .net     -> dotnet
            node.js  -> nodejs
            react.js -> reactjs
        """
        token = str(token).strip().lower()

        if not token:
            return ""

        if token in TECH_TOKEN_VARIANTS:
            return TECH_TOKEN_VARIANTS[token]

        token = token.strip(".,;:()[]{}")

        if token in TECH_TOKEN_VARIANTS:
            return TECH_TOKEN_VARIANTS[token]

        return token

    def _build_idf_scores(self, documents_tokens: list[list[str]]) -> dict[str, float]:
        """
        Build smoothed IDF scores from the job-description corpus.

        IDF formula:
            idf = log((N + 1) / (df + 1)) + 1

        This gives higher weight to tokens that appear in fewer job descriptions.
        """
        if not self.use_idf_weighting:
            return {}

        total_docs = len(documents_tokens)

        if total_docs == 0:
            return {}

        document_frequency: Counter[str] = Counter()

        for tokens in documents_tokens:
            document_frequency.update(set(tokens))

        idf_scores: dict[str, float] = {}

        for token, df in document_frequency.items():
            idf_scores[token] = math.log((total_docs + 1) / (df + 1)) + 1.0

        return idf_scores

    def _document_vector(
        self,
        tokens: list[str],
        vectors: Any,
        idf_scores: dict[str, float],
    ) -> np.ndarray | None:
        """
        Convert a token list into one document vector.

        Instead of simple mean pooling, this method uses weighted average:

            document_vector = average(token_vector * token_weight)

        where:
            token_weight = IDF(token) * log-scaled term frequency

        This reduces the impact of generic repeated terms and gives more
        importance to discriminative terms such as python, sql, finance,
        accounting, logistics, healthcare, react, etc.
        """
        if not tokens:
            return None

        token_counts = Counter(tokens)

        token_vectors: list[np.ndarray] = []
        token_weights: list[float] = []

        for token, count in token_counts.items():
            vector = self._lookup_vector(token, vectors)

            if vector is None:
                continue

            tf_weight = 1.0 + math.log(float(count))

            if self.use_idf_weighting:
                idf_weight = idf_scores.get(token, 1.0)
            else:
                idf_weight = 1.0

            weight = tf_weight * idf_weight

            token_vectors.append(vector)
            token_weights.append(weight)

        if not token_vectors:
            return None

        matrix = np.vstack(token_vectors).astype(np.float32)
        weights = np.asarray(token_weights, dtype=np.float32)

        if float(weights.sum()) == 0.0:
            return None

        document_vector = np.average(
            matrix,
            axis=0,
            weights=weights,
        ).astype(np.float32)

        if float(np.linalg.norm(document_vector)) == 0.0:
            return None

        return document_vector

    def _lookup_vector(self, token: str, vectors: Any) -> np.ndarray | None:
        """
        Look up a token vector using multiple candidate forms.

        GloVe vocabulary is usually lowercase, but this method is kept robust.
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

        if token.lower() in TECH_TOKEN_VARIANTS:
            candidates.append(TECH_TOKEN_VARIANTS[token.lower()])

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
        Do not use (cosine + 1) / 2 because that artificially inflates scores.
        """
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

        if denominator == 0.0:
            return 0.0

        cosine_value = float(np.dot(a, b) / denominator)

        return float(np.clip(cosine_value, 0.0, 1.0))