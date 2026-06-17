import pickle
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# Load .env from project root, not from scripts/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

# DATA_PATH = PROJECT_ROOT / "data" / "processed" / "resume_chunk_embeddings.pkl"
# COLLECTION_NAME = "resume_embeddings"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "job_description_chunk_embeddings.pkl"
COLLECTION_NAME = "job_description_embeddings"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("ENV_PATH =", ENV_PATH)
print("QDRANT_URL =", QDRANT_URL)
print("QDRANT_API_KEY exists =", bool(QDRANT_API_KEY))

if not QDRANT_URL:
    raise ValueError(
        "QDRANT_URL is missing. Please set QDRANT_URL in .env."
    )

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY or None,
)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

# texts = data["texts"]
# categories = data["categories"]
# embeddings = data["embeddings"]

# resume_ids = data.get("resume_ids", [f"resume_{i}" for i in range(len(texts))])
# chunk_ids = data.get("chunk_ids", [f"chunk_{i}" for i in range(len(texts))])
# chunk_indexes = data.get("chunk_indexes", list(range(len(texts))))

# print("Loaded data:")
# print("texts:", len(texts))
# print("categories:", len(categories))
# print("embeddings:", embeddings.shape)
# print("resume_ids:", len(resume_ids))
# print("chunk_ids:", len(chunk_ids))
# print("chunk_indexes:", len(chunk_indexes))


texts = data["texts"]
job_ids = data["job_id"]
chunk_ids = data["chunk_id"]
chunk_indexes = data["chunk_index"]
job_titles = data["job_title"]
categories = data["category"]
required_skills = data["required_skills"]
job_descriptions = data["job_description"]
embeddings = data["embeddings"]

# client = QdrantClient(url="http://localhost:6333")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=120,
)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)

points = [
    PointStruct(
        id=i,
        vector=embeddings[i].tolist(),
        payload={
            "job_id": str(job_ids[i]),
            "chunk_id": str(chunk_ids[i]),
            "chunk_index": int(chunk_indexes[i]),
            "job_title": str(job_titles[i]),
            "category": str(categories[i]),
            "required_skills": str(required_skills[i]),
            "job_description": str(job_descriptions[i]),
            "text": str(texts[i]),
        }
    )
    for i in range(len(texts))
]

BATCH_SIZE = 32

for start in range(0, len(points), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = points[start:end]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch,
        wait=True,
    )

    print(f"Inserted {min(end, len(points))}/{len(points)} points")

count = client.count(collection_name=COLLECTION_NAME).count
print(f"Inserted {count} points into Qdrant.")