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
    k1: float = 1.5
    b: float = 0.75
    name: str = "BM25"

    def score(self, resume_text: str, job_texts: list[str]) -> list[float]:
        """
        Score multiple job descriptions against one resume.

        Input:
        - resume_text: one candidate CV/resume
        - job_texts: list of job descriptions

        Output:
        - normalized BM25 score for each job description
        """
        if not job_texts:
            return []

        tokenized_jobs = [tokenize_words(text) for text in job_texts]

        # CV is the query
        query_tokens = list(dict.fromkeys(tokenize_words(resume_text)))

        if not query_tokens or not any(tokenized_jobs):
            return [0.0 for _ in job_texts]

        raw_scores = SimpleBM25(
            tokenized_jobs,
            k1=self.k1,
            b=self.b
        ).get_scores(query_tokens)

        return self._normalize(raw_scores, query_length=len(query_tokens))

    @staticmethod
    def _normalize(scores: list[float] | np.ndarray, query_length: int = 1) -> list[float]:
        array = np.asarray(scores, dtype=float)
        if array.size == 0:
            return []

        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        array = np.maximum(array, 0.0)

        if math.isclose(float(array.max()), 0.0):
            return [0.0 for _ in array]

        scale = max(float(query_length), 1.0)
        normalized = 1.0 - np.exp(-array / scale)

        return np.clip(normalized, 0.0, 1.0).astype(float).tolist()