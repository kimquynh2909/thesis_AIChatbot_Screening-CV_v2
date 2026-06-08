from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_for_matching


@dataclass
class TfidfMatcher:
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 30000

    name: str = "TF-IDF + Cosine Similarity"

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        corpus = [clean_for_matching(jd_text)] + [clean_for_matching(text) for text in resume_texts]
        if not any(text.strip() for text in corpus):
            return [0.0 for _ in resume_texts]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return np.clip(scores, 0.0, 1.0).astype(float).tolist()
