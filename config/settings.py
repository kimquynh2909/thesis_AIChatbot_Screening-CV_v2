from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # The app still runs with environment variables if python-dotenv is absent.
    load_dotenv = None

if load_dotenv:
    load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_RESUME_DIR = DATA_DIR / "sample_resumes"
SAMPLE_JD_DIR = DATA_DIR / "sample_job_descriptions"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation_results"
CHART_OUTPUT_DIR = OUTPUT_DIR / "charts"
EXPORTED_REPORT_DIR = OUTPUT_DIR / "exported_reports"
SKILL_DICTIONARY_PATH = PROJECT_ROOT / "config" / "skill_dictionary.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

MODEL_NAMES = {
    "tfidf": "TF-IDF + Cosine Similarity",
    "bm25": "BM25",
    "word2vec": "Word2Vec Average Embeddings",
    "glove": "GloVe Average Embeddings",
    "sbert": "SBERT",
    "e5": "E5",
    "bge": "BGE",
    "google": "Google Embeddings",
}

DEFAULT_EMBEDDING_MODELS = {
    "word2vec": os.getenv("WORD2VEC_MODEL", "word2vec-google-news-300"),
    "glove": os.getenv("GLOVE_MODEL", "glove-wiki-gigaword-100"),
    "sbert": os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    "e5": os.getenv("E5_MODEL", "intfloat/e5-base-v2"),
    "bge": os.getenv("BGE_MODEL", "BAAI/bge-base-en-v1.5"),
    "google": os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
}

STATIC_EMBEDDING_PATHS = {
    "word2vec": "data/embeddings/GoogleNews-vectors-negative300.bin",
    "glove": os.getenv("GLOVE_PATH") or "data/embeddings/glove.6B.100d.txt",
}

DEFAULT_RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODELS["e5"])
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "resume_rag_chunks")
QDRANT_LOCAL_PATH = Path(os.getenv("QDRANT_LOCAL_PATH", str(OUTPUT_DIR / "qdrant_local")))

DEFAULT_HYBRID_WEIGHTS = {
    "semantic": 0.50,
    "skills": 0.30,
    "experience": 0.10,
    "education": 0.10,
}

RECOMMENDATION_THRESHOLDS = {
    "strong": 75.0,
    "potential": 55.0,
}

PRIMARY_KAGGLE_DATASETS = {
    "ranking_pairs": "thejohnwick001/resume-data-for-ranking",
    "labeled_synthetic": "surendra365/recruitement-dataset",
    "structured_workflow": "suvroo/ai-powered-job-application-screening-system",
    "job_descriptions": "adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles",
    "resume_livecareer": "snehaanbhawal/resume-dataset",
    "job_skill_set": "batuhanmutlu/job-skill-set",
}

RANDOM_SEED = 42
DEFAULT_TOP_K = 10
