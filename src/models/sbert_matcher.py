from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import util

from config.settings import DEFAULT_EMBEDDING_MODELS
from src.preprocessing.text_cleaner import clean_for_matching


@lru_cache(maxsize=2)
def load_sbert_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for SBERT. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer(model_name)


@dataclass
class SBERTMatcher:
    model_name: str = DEFAULT_EMBEDDING_MODELS["sbert"]
    name: str = "SBERT"
    normalize_embeddings: bool = True

    def score(self, jd_text: str, resume_texts: list[str]) -> list[float]:
        """
        Score one job description against multiple resumes.

        Uses the same matching preprocessing as TF-IDF:
        clean_for_matching(jd_text) and clean_for_matching(each resume).
        """
        if not resume_texts:
            return []

        jd_clean = clean_for_matching(jd_text)
        resume_clean = [clean_for_matching(text) for text in resume_texts]

        if not jd_clean.strip():
            return [0.0 for _ in resume_clean]

        non_empty_indexes = [
            index for index, text in enumerate(resume_clean) if text.strip()
        ]
        if not non_empty_indexes:
            return [0.0 for _ in resume_clean]

        model = load_sbert_model(self.model_name)
        jd_embedding = self._encode(model, [jd_clean])[0]
        resume_embeddings = self._encode(
            model,
            [resume_clean[index] for index in non_empty_indexes],
        )

        raw_scores = util.cos_sim(jd_embedding, resume_embeddings)[0].tolist()
        scores = [0.0 for _ in resume_clean]
        for index, raw_score in zip(non_empty_indexes, raw_scores, strict=False):
            scores[index] = float(np.clip(raw_score, 0.0, 1.0))

        return scores

    def _encode(self, model, texts: list[str]) -> np.ndarray:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float)
