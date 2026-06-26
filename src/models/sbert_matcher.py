from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from rich import _console
from sentence_transformers import util

from config.settings import DEFAULT_EMBEDDING_MODELS
from src.llm.structured_extraction import (
    extract_cv_json,
    extract_job_description_json,
)
from src.preprocessing.skill_extractor import SkillExtractor
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
    """
    SBERT matcher for CV-to-job matching.

    Project direction:
        one CV/resume -> many job descriptions

    Supported modes:
    1. Raw text mode:
        clean text -> SBERT embedding -> cosine similarity

    2. Structured extraction mode:
        LLM/deterministic field extraction
        -> structured text
        -> SBERT embedding
        -> cosine similarity

    Public method:
        score(resume_text, job_texts)
    """

    model_name: str = DEFAULT_EMBEDDING_MODELS["sbert"]
    name: str = "SBERT CV-to-Job Similarity"
    normalize_embeddings: bool = True

    use_structured_extraction: bool = False
    use_llm_extraction: bool = False
    include_raw_text: bool = False

    debug: bool = True  
    debug_max_chars: int = 12000

    skill_extractor: SkillExtractor | None = None

    def score(self, resume_text: str, job_texts: list[str]) -> list[float]:
        """
        Score one CV/resume against many job descriptions.

        Args:
            resume_text:
                One CV/resume text.

            job_texts:
                List of job description texts.

        Returns:
            List of scores in range [0, 1].
            The order of scores follows the order of job_texts.
        """
        if not job_texts:
            return []

        if self.use_structured_extraction:
            return self._score_with_structured_extraction(
                resume_text=resume_text,
                job_texts=job_texts,
            )

        return self._score_with_raw_text(
            resume_text=resume_text,
            job_texts=job_texts,
        )

    def _score_with_raw_text(
        self,
        resume_text: str,
        job_texts: list[str],
    ) -> list[float]:
        """
        Raw text flow:
            CV text -> clean_for_matching()
            Job texts -> clean_for_matching()
            SBERT cosine similarity
        """
        resume_clean = clean_for_matching(resume_text)
        job_clean_texts = [clean_for_matching(text) for text in job_texts]

        if not resume_clean.strip():
            return [0.0 for _ in job_texts]

        non_empty_indexes = [
            index
            for index, text in enumerate(job_clean_texts)
            if text.strip()
        ]

        if not non_empty_indexes:
            return [0.0 for _ in job_texts]

        model = load_sbert_model(self.model_name)

        resume_embedding = self._encode(model, [resume_clean])[0]

        selected_job_texts = [
            job_clean_texts[index]
            for index in non_empty_indexes
        ]

        job_embeddings = self._encode(model, selected_job_texts)

        raw_scores = util.cos_sim(
            resume_embedding,
            job_embeddings,
        )[0].tolist()

        scores = [0.0 for _ in job_texts]

        for index, raw_score in zip(non_empty_indexes, raw_scores, strict=False):
            scores[index] = self._clip_score(raw_score)

        return scores

    def _score_with_structured_extraction(
        self,
        resume_text: str,
        job_texts: list[str],
    ) -> list[float]:
        """
        Structured extraction flow:
            For each job:
                1. Extract JD fields.
                2. Extract CV fields using JD extraction.
                3. Convert both JSON extractions into structured text.
                4. Compute SBERT cosine similarity.

        This is more explainable but slower than raw text mode.
        """
        if not resume_text or not resume_text.strip():
            return [0.0 for _ in job_texts]

        extractor = self.skill_extractor or SkillExtractor()

        job_matching_texts: list[str] = []
        resume_matching_texts: list[str] = []

        for job_text in job_texts:
            if not isinstance(job_text, str) or not job_text.strip():
                job_matching_texts.append("")
                resume_matching_texts.append("")
                continue

            jd_extraction = extract_job_description_json(
                jd_text=job_text,
                use_llm=self.use_llm_extraction,
                skill_extractor=extractor,
            )

            cv_extraction = extract_cv_json(
                resume_text=resume_text,
                jd_extraction=jd_extraction,
                use_llm=self.use_llm_extraction,
                skill_extractor=extractor,
            )

            job_matching_text = self._build_jd_matching_text(
                raw_text=job_text,
                extraction=jd_extraction,
            )

            resume_matching_text = self._build_cv_matching_text(
                raw_text=resume_text,
                extraction=cv_extraction,
            )

            self._debug_print("Structured JD text for SBERT", job_matching_text)
            self._debug_print("Structured CV text for SBERT", resume_matching_text)

            job_matching_texts.append(job_matching_text)
            resume_matching_texts.append(resume_matching_text)

        non_empty_indexes = [
            index
            for index, job_text in enumerate(job_matching_texts)
            if job_text.strip() and resume_matching_texts[index].strip()
        ]

        if not non_empty_indexes:
            return [0.0 for _ in job_texts]

        selected_job_texts = [
            job_matching_texts[index]
            for index in non_empty_indexes
        ]

        selected_resume_texts = [
            resume_matching_texts[index]
            for index in non_empty_indexes
        ]

        model = load_sbert_model(self.model_name)

        job_embeddings = self._encode(model, selected_job_texts)
        resume_embeddings = self._encode(model, selected_resume_texts)

        similarity_matrix = util.cos_sim(
            resume_embeddings,
            job_embeddings,
        )

        diagonal_scores = similarity_matrix.diag().tolist()

        scores = [0.0 for _ in job_texts]

        for index, raw_score in zip(non_empty_indexes, diagonal_scores, strict=False):
            scores[index] = self._clip_score(raw_score)

        return scores

    def _build_jd_matching_text(
        self,
        raw_text: str,
        extraction: dict[str, Any],
    ) -> str:
        """
        Convert extracted JD JSON into structured text for SBERT.
        """
        parts = [
            self._field_to_text(
                "required skills",
                extraction.get("required_skills", []),
            ),
            self._field_to_text(
                "preferred skills",
                extraction.get("preferred_skills", []),
            ),
            self._field_to_text(
                "tools",
                extraction.get("tools", []),
            ),
            self._field_to_text(
                "education requirements",
                extraction.get("education", []),
            ),
            self._field_to_text(
                "experience requirements",
                extraction.get("experience", []),
            ),
        ]

        structured_text = "\n".join(
            part
            for part in parts
            if part.strip()
        )

        if self.include_raw_text:
            raw_clean = clean_for_matching(raw_text)
            structured_text = (
                f"{structured_text}\n"
                f"raw job description: {raw_clean}"
            )

        return structured_text.strip()

    def _build_cv_matching_text(
        self,
        raw_text: str,
        extraction: dict[str, Any],
    ) -> str:
        """
        Convert extracted CV JSON into structured text for SBERT.
        """
        parts = [
            self._field_to_text(
                "skills",
                extraction.get("skills", []),
            ),
            self._field_to_text(
                "education",
                extraction.get("education", []),
            ),
            self._field_to_text(
                "experience",
                extraction.get("experience", []),
            ),
            self._field_to_text(
                "projects",
                extraction.get("projects", []),
            ),
        ]

        structured_text = "\n".join(
            part
            for part in parts
            if part.strip()
        )

        if self.include_raw_text:
            raw_clean = clean_for_matching(raw_text)
            structured_text = (
                f"{structured_text}\n"
                f"raw resume: {raw_clean}"
            )

        return structured_text.strip()

    @staticmethod
    def _field_to_text(label: str, value: Any) -> str:
        items = SBERTMatcher._string_list(value)

        if not items:
            return ""

        return f"{label}: {', '.join(items)}"

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        """
        Normalize string/list/dict values into a clean list of strings.
        """
        if value is None:
            return []

        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []

        if isinstance(value, list):
            result: list[str] = []

            for item in value:
                if isinstance(item, dict):
                    text = (
                        item.get("name")
                        or item.get("requirement")
                        or item.get("text")
                        or item.get("value")
                    )

                    if text:
                        result.append(str(text).strip())

                else:
                    text = str(item).strip()

                    if text:
                        result.append(text)

            return result

        text = str(value).strip()
        return [text] if text else []

    def _encode(self, model, texts: list[str]) -> np.ndarray:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        return np.asarray(embeddings, dtype=float)

    @staticmethod
    def _clip_score(score: float) -> float:
        return float(np.clip(float(score), 0.0, 1.0))
    
    def _debug_print(self, title: str, text: str) -> None:
        if not getattr(self, "debug", False):
            return

        max_chars = getattr(self, "debug_max_chars", 1200)

        print("\n" + "=" * 80, flush=True)
        print(title, flush=True)
        print("-" * 80, flush=True)

        text = str(text)

        if len(text) > max_chars:
            print(text[:max_chars], flush=True)
            print(f"\n... [truncated, total chars: {len(text)}]", flush=True)
        else:
            print(text, flush=True)

        print("=" * 80 + "\n", flush=True)
