from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]

word2vec_path = PROJECT_ROOT / "data" / "embeddings" / "GoogleNews-vectors-negative300.bin"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text: str, preserve_case: bool = False) -> str:
    text = normalize_unicode(strip_html(text))
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[-=_]{4,}", " ", text)
    text = re.sub(r"[^\w\s.,;:()&/#%+\-@]", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = text.strip()

    return text if preserve_case else text.lower()


def clean_for_matching(text: str) -> str:
    return clean_text(text, preserve_case=False)


def tokenize_like_vscode(text: str, min_token_length: int = 2) -> list[str]:
    cleaned_text = clean_for_matching(str(text))

    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9+#.\-]*",
        cleaned_text,
    )

    return [
        token
        for token in tokens
        if len(token.strip()) >= min_token_length
    ]


def token_candidates(token: str) -> list[str]:
    token = str(token).strip()

    candidates = [
        token,
        token.lower(),
        token.title(),
        token.upper(),
    ]

    normalized_variants = {
        "c++": "cpp",
        "c#": "csharp",
        ".net": "dotnet",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "vue.js": "vuejs",
        "next.js": "nextjs",
    }

    lower = token.lower()

    if lower in normalized_variants:
        candidates.append(normalized_variants[lower])

    unique_candidates: list[str] = []

    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates


def avg_feature_vector(
    sentence: str,
    model: KeyedVectors,
    num_features: int = 300,
) -> tuple[np.ndarray, list[str], list[str]]:
    words = tokenize_like_vscode(str(sentence))

    token_vectors: list[np.ndarray] = []
    matched_tokens: list[str] = []
    missing_tokens: list[str] = []

    for word in words:
        found_vector = None
        found_token = None

        for candidate in token_candidates(word):
            if candidate in model.key_to_index:
                found_vector = model[candidate]
                found_token = candidate
                break

        if found_vector is not None:
            token_vectors.append(found_vector)
            matched_tokens.append(str(found_token))
        else:
            missing_tokens.append(word)

    if not token_vectors:
        return np.zeros((num_features,), dtype="float32"), matched_tokens, missing_tokens

    feature_vec = np.mean(
        np.vstack(token_vectors),
        axis=0,
    ).astype("float32")

    return feature_vec, matched_tokens, missing_tokens


def main() -> None:
    if not word2vec_path.exists():
        raise FileNotFoundError(f"Word2Vec file not found: {word2vec_path}")

    print("Loading Word2Vec from:")
    print(word2vec_path)

    model = KeyedVectors.load_word2vec_format(
        str(word2vec_path),
        binary=True,
    )

    print("Vector size:", model.vector_size)
    print("Vocabulary size:", len(model.key_to_index))
    print("king queen similarity:", model.similarity("king", "queen"))

    jd_text = (
        "A Fitness Coach is responsible for helping clients achieve their fitness goals "
        "by designing and leading group or individual fitness programs. You will provide "
        "instruction on exercises, proper form, and injury prevention techniques, encouraging "
        "clients to push their limits while maintaining a focus on their well-being. The role "
        "requires a passion for health and fitness, a strong understanding of exercise physiology, "
        "and the ability to motivate and inspire others. You will also monitor clients’ progress "
        "and make adjustments to their fitness plans as needed to ensure continuous improvement."
    )

    resume_text = (
        "Proficient in Injury Prevention, Motivation, Nutrition, Health Coaching, Strength Training, "
        "with mid-level experience in the field. Holds a Bachelors degree. Holds certifications such "
        "as Certified Personal Trainer (CPT) by NASM. Skilled in delivering results and adapting to "
        "dynamic environments."
    )

    jd_vec, jd_matched, jd_missing = avg_feature_vector(jd_text, model)
    resume_vec, resume_matched, resume_missing = avg_feature_vector(resume_text, model)

    score = cosine_similarity(
        jd_vec.reshape(1, -1),
        resume_vec.reshape(1, -1),
    )[0][0]

    score = float(np.clip(score, 0.0, 1.0))

    print("\nMatching score:", score)

    print("\nJD matched:", len(jd_matched))
    print("JD missing:", len(jd_missing))
    print("JD matched sample:", jd_matched[:50])
    print("JD missing sample:", jd_missing[:50])

    print("\nResume matched:", len(resume_matched))
    print("Resume missing:", len(resume_missing))
    print("Resume matched sample:", resume_matched[:50])
    print("Resume missing sample:", resume_missing[:50])

    print("\nJD vector first 10:", jd_vec[:10])
    print("Resume vector first 10:", resume_vec[:10])
    print("JD norm:", np.linalg.norm(jd_vec))
    print("Resume norm:", np.linalg.norm(resume_vec))


if __name__ == "__main__":
    main()