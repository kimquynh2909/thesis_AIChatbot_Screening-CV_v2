import numpy as np
from sentence_transformers import SentenceTransformer, util

resume_text = """
Proficient in Injury Prevention, Motivation, Nutrition, Health Coaching, Strength Training,
with mid-level experience in the field. Holds a Bachelors degree. Holds certifications such as
Certified Personal Trainer (CPT) by NASM. Skilled in delivering results and adapting to dynamic environments.
"""

jd_text = """
A Fitness Coach is responsible for helping clients achieve their fitness goals by designing and leading
group or individual fitness programs. You will provide instruction on exercises, proper form, and injury
prevention techniques, encouraging clients to push their limits while maintaining a focus on their well-being.
The role requires a passion for health and fitness, a strong understanding of exercise physiology, and the
ability to motivate and inspire others. You will also monitor clients’ progress and make adjustments to their
fitness plans as needed to ensure continuous improvement.
"""

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

jd_embedding = model.encode(
    jd_text,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

resume_embedding = model.encode(
    resume_text,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

score = util.cos_sim(jd_embedding, resume_embedding).item()

match_percentage = np.clip(score, 0, 1) * 100

print(f"SBERT score       : {score:.4f}")
print(f"Match percentage : {match_percentage:.2f}%")