from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.preprocessing.text_cleaner import tokenize_words


class SimpleBM25:
    """Small fallback BM25 implementation used when rank-bm25 is unavailable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_freq = Counter()
        self.term_freqs = [Counter(doc) for doc in corpus_tokens]
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        for doc in corpus_tokens:
            self.doc_freq.update(set(doc))
        self.n_docs = len(corpus_tokens)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx] or 1
            score = 0.0
            for term in query_tokens:
                if term not in term_freq:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                tf = term_freq[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1))
                score += idf * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


@dataclass
class BM25Matcher:
    name: str = "BM25"

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        tokenized_resumes = [tokenize_words(text) for text in resume_texts]
        query_tokens = tokenize_words(jd_text)
        if not query_tokens or not any(tokenized_resumes):
            return [0.0 for _ in resume_texts]

        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokenized_resumes)
            raw_scores = bm25.get_scores(query_tokens)
        except ImportError:
            raw_scores = SimpleBM25(tokenized_resumes).get_scores(query_tokens)

        return self._normalize(raw_scores)

    @staticmethod
    def _normalize(scores: list[float] | np.ndarray) -> list[float]:
        array = np.asarray(scores, dtype=float)
        if array.size == 0:
            return []
        max_score = float(array.max())
        min_score = float(array.min())
        if math.isclose(max_score, min_score):
            return [0.0 for _ in array]
        normalized = (array - min_score) / (max_score - min_score)
        return np.clip(normalized, 0.0, 1.0).astype(float).tolist()
