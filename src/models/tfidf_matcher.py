from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_for_matching


@dataclass
class TfidfMatcher:
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 30000

    name: str = "TF-IDF"
    vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)

    def fit(self, corpus: Iterable[str]) -> "TfidfMatcher":
        """
        Fit TF-IDF once on a benchmark/training corpus.

        If this method is not called, score() preserves the interactive fallback
        behavior and fits on the one JD plus the provided resumes.
        """
        cleaned_corpus = [
            clean_for_matching(text)
            for text in corpus
            if str(text).strip()
        ]
        if not cleaned_corpus:
            self.vectorizer = None
            return self

        vectorizer = self._new_vectorizer()
        try:
            vectorizer.fit(cleaned_corpus)
        except ValueError:
            self.vectorizer = None
            return self
        self.vectorizer = vectorizer
        return self

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        corpus = [clean_for_matching(jd_text)] + [clean_for_matching(text) for text in resume_texts]
        if not any(text.strip() for text in corpus):
            return [0.0 for _ in resume_texts]

        if self.vectorizer is not None:
            matrix = self.vectorizer.transform(corpus)
        else:
            vectorizer = self._new_vectorizer()
            try:
                matrix = vectorizer.fit_transform(corpus)
            except ValueError:
                return [0.0 for _ in resume_texts]

        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return np.clip(scores, 0.0, 1.0).astype(float).tolist()

    def _new_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=1,
        )
