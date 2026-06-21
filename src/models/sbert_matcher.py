import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

resume_path = "data/processed/train_resumes.xlsx"

jd_text = """
We are looking for a Data Scientist with experience in Python, SQL,
machine learning, data preprocessing, model evaluation, statistics,
and communication skills.
"""

df = pd.read_excel(resume_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

jd_embedding = model.encode(
    jd_text,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

resume_embeddings = model.encode(
    df["text"].fillna("").astype(str).tolist(),
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

scores = util.cos_sim(jd_embedding, resume_embeddings)[0].cpu().numpy()

df["sbert_score"] = scores
df["match_percentage"] = (np.clip(scores, 0, 1) * 100).round(2)

df = df.sort_values("sbert_score", ascending=False).reset_index(drop=True)
df.insert(0, "rank", range(1, len(df) + 1))

print(df[["rank", "category", "sbert_score", "match_percentage"]].head(10))

df.to_excel("outputs/sbert_result.xlsx", index=False)