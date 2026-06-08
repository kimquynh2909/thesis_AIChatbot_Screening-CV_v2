from __future__ import annotations

MODEL_TFIDF = "tfidf"
MODEL_BM25 = "bm25"
MODEL_WORD2VEC = "word2vec"
MODEL_GLOVE = "glove"
MODEL_SBERT = "sbert"
MODEL_E5 = "e5"
MODEL_BGE = "bge"
MODEL_GOOGLE = "google"

SUPPORTED_MODEL_KEYS = [
    MODEL_TFIDF,
    MODEL_BM25,
    MODEL_WORD2VEC,
    MODEL_GLOVE,
    MODEL_SBERT,
    MODEL_E5,
    MODEL_BGE,
    MODEL_GOOGLE,
]

RELEVANCE_LABEL_THRESHOLD = 0.65

TEXT_PREVIEW_CHARS = 1200

DEGREE_KEYWORDS = [
    "bachelor",
    "bs",
    "bsc",
    "ba",
    "master",
    "ms",
    "msc",
    "mba",
    "phd",
    "doctorate",
    "degree",
    "computer science",
    "information technology",
    "software engineering",
    "data science",
]

CERTIFICATION_KEYWORDS = [
    "certified",
    "certification",
    "aws certified",
    "azure certified",
    "google cloud certified",
    "pmp",
    "scrum master",
    "cissp",
    "comptia",
    "oracle certified",
    "microsoft certified",
]

EXPERIENCE_PATTERNS = [
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?experience",
    r"experience\s+(?:of\s+)?(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    r"(?P<years>\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+in",
]
