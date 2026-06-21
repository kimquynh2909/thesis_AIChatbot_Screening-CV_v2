# from __future__ import annotations

# from dataclasses import dataclass
# from functools import lru_cache

# import numpy as np

# from src.preprocessing.chunker import aggregate_chunk_scores, chunk_text
# from src.preprocessing.text_cleaner import clean_for_matching


# def cosine(a: np.ndarray, b: np.ndarray) -> float:
#     denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
#     if denominator == 0:
#         return 0.0
#     return float(np.dot(a, b) / denominator)


# @lru_cache(maxsize=4)
# def load_sentence_transformer(model_name: str):
#     try:
#         from sentence_transformers import SentenceTransformer
#     except ImportError as exc:
#         raise ImportError(
#             "sentence-transformers is required for SBERT, E5, and BGE models. "
#             "Install dependencies with: pip install -r requirements.txt"
#         ) from exc
#     return SentenceTransformer(model_name)


# @dataclass
# class SentenceEmbeddingMatcher:
#     model_name: str
#     name: str
#     query_prefix: str = ""
#     passage_prefix: str = ""
#     normalize_embeddings: bool = True
#     max_words: int = 220
#     overlap_words: int = 40

#     def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
#         if not resume_texts:
#             return []
#         model = load_sentence_transformer(self.model_name)

#         jd_clean = clean_for_matching(jd_text)
#         jd_embedding = self._encode(model, [self.query_prefix + jd_clean])[0]

#         scores: list[float] = []
#         for resume_text in resume_texts:
#             chunks = chunk_text(clean_for_matching(resume_text), self.max_words, self.overlap_words)
#             if not chunks:
#                 scores.append(0.0)
#                 continue
#             chunk_texts = [self.passage_prefix + chunk.text for chunk in chunks]
#             chunk_embeddings = self._encode(model, chunk_texts)
#             chunk_scores = [self._cosine_01(jd_embedding, embedding) for embedding in chunk_embeddings]
#             scores.append(aggregate_chunk_scores(chunk_scores, strategy="mean_top3"))
#         return scores

#     def _encode(self, model, texts: list[str]) -> np.ndarray:
#         embeddings = model.encode(
#             texts,
#             convert_to_numpy=True,
#             normalize_embeddings=self.normalize_embeddings,
#             show_progress_bar=False,
#         )
#         return np.asarray(embeddings, dtype=float)

#     @staticmethod
#     def _cosine_01(a: np.ndarray, b: np.ndarray) -> float:
#         value = cosine(a, b)
#         return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
