from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from config.settings import SKILL_DICTIONARY_PATH
from src.preprocessing.text_cleaner import clean_for_matching


@dataclass(frozen=True)
class SkillAnalysis:
    jd_skills: list[str]
    resume_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    skill_score: float


class SkillExtractor:
    def __init__(self, dictionary_path: Path = SKILL_DICTIONARY_PATH):
        self.dictionary_path = dictionary_path
        self.skills_by_category = self._load_dictionary(dictionary_path)
        self.skill_aliases = self._flatten_skills(self.skills_by_category)

    @staticmethod
    def _load_dictionary(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            raise FileNotFoundError(f"Skill dictionary not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return {str(category): [str(skill).lower() for skill in skills] for category, skills in raw.items()}

    @staticmethod
    def _flatten_skills(skills_by_category: dict[str, list[str]]) -> list[str]:
        skills = {skill.strip().lower() for skills in skills_by_category.values() for skill in skills}
        return sorted(skills, key=lambda item: (-len(item), item))

    def extract(self, text: str) -> list[str]:
        cleaned = f" {clean_for_matching(text)} "
        found: set[str] = set()
        for skill in self.skill_aliases:
            pattern = self._skill_pattern(skill)
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                found.add(skill)
        return sorted(found)

    @staticmethod
    def _skill_pattern(skill: str) -> str:
        escaped = re.escape(skill)
        if re.search(r"[+#.]", skill):
            return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        return rf"\b{escaped}\b"

    def analyze(self, resume_text: str, jd_text: str) -> SkillAnalysis:
        jd_skills = self.extract(jd_text)
        resume_skills = self.extract(resume_text)
        matched = sorted(set(jd_skills).intersection(resume_skills))
        missing = sorted(set(jd_skills).difference(resume_skills))
        score = len(matched) / len(jd_skills) if jd_skills else 0.0
        return SkillAnalysis(
            jd_skills=jd_skills,
            resume_skills=resume_skills,
            matched_skills=matched,
            missing_skills=missing,
            skill_score=float(score),
        )
