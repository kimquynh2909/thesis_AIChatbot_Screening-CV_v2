from __future__ import annotations

from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import clean_for_matching


def test_clean_text_preserves_technical_tokens() -> None:
    text = clean_for_matching("Python, C++, C#, REST API, AWS!!!")
    assert "python" in text
    assert "c++" in text
    assert "c#" in text
    assert "rest api" in text


def test_skill_extractor_finds_overlap() -> None:
    extractor = SkillExtractor()
    analysis = extractor.analyze(
        resume_text="Python developer with SQL, Docker, AWS and scikit-learn experience.",
        jd_text="Need Python, SQL, AWS, Kubernetes and PyTorch.",
    )
    assert {"python", "sql", "aws"}.issubset(set(analysis.matched_skills))
    assert "kubernetes" in analysis.missing_skills
    assert 0 < analysis.skill_score < 1


def test_document_parser_txt_bytes() -> None:
    parsed = DocumentParser().parse(b"Resume\nSkills: Python and SQL", filename="resume.txt")
    assert parsed.extension == ".txt"
    assert "Python" in parsed.cleaned_text
