# 🎯 Quick Reference - Terminal Commands

## Cài đặt (Setup)
```bash
cd ai_resume_screening

# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tạo dữ liệu mẫu
python scripts/create_sample_data.py
```

---

## Chạy Ứng Dụng

### 📱 Streamlit (Web UI)
```bash
streamlit run app.py
# Truy cập: http://localhost:8501
```

### 💻 Terminal/CLI (Không cần Streamlit)
```bash
# Mặc định (SBERT + sample data)
python cli_screening.py

# Với model khác
python cli_screening.py --model e5
python cli_screening.py --model bge
python cli_screening.py --model bm25
python cli_screening.py --model tfidf

# Với file riêng
python cli_screening.py --jd jobs/job.txt --resumes resumes/cv1.txt resumes/cv2.txt

# Chi tiết + lưu CSV
python cli_screening.py --verbose --output results.csv

# JSON output
python cli_screening.py --json > results.json

# Help
python cli_screening.py --help
```

---

## 📊 Evaluation & Testing

### Benchmark models
```bash
python -m src.evaluation.benchmark \
  --pairs data/processed/test_pairs.csv \
  --models tfidf,bm25,e5,bge
```

### Run tests
```bash
pytest
pytest -v  # verbose
pytest tests/test_matching.py  # specific test
```

### Index to Qdrant (vector database)
```bash
python scripts/index_qdrant.py

# Custom files
python scripts/index_qdrant.py \
  --jd data/sample_job_descriptions/job.txt \
  --resumes data/sample_resumes/*.txt
```

---

## 🔧 Common Use Cases

### 1️⃣ Nhanh chóng qua lô candidate
```bash
python cli_screening.py \
  --jd jobs/position.txt \
  --resumes candidates/*.txt \
  --model tfidf
```

### 2️⃣ Chi tiết + lưu kết quả
```bash
python cli_screening.py \
  --jd jobs/senior_dev.txt \
  --resumes team/*.txt \
  --model e5 \
  --verbose \
  --output screening_results.csv
```

### 3️⃣ So sánh nhiều model
```bash
for model in tfidf bm25 e5 bge; do
  echo "Testing $model..."
  python cli_screening.py --model $model --output results_$model.csv
done
```

### 4️⃣ Semantic similarity only (no hybrid)
```bash
python cli_screening.py \
  --semantic 1.0 --skills 0 --experience 0 --education 0 \
  --model bge
```

### 5️⃣ Tùy chỉnh weights
```bash
python cli_screening.py \
  --semantic 0.6 --skills 0.3 --experience 0.07 --education 0.03
```

---

## 📂 Project Structure

```
ai_resume_screening/
├── cli_screening.py          ← ✨ NEW: Terminal interface
├── app.py                    ← Streamlit web app
├── CLI_GUIDE.md             ← ✨ NEW: Detailed guide
├── requirements.txt
├── config/
│   └── settings.py          ← Model configurations
├── src/
│   ├── models/
│   │   ├── tfidf_matcher.py
│   │   ├── bm25_matcher.py
│   │   ├── sbert_matcher.py
│   │   ├── e5_matcher.py
│   │   ├── bge_matcher.py
│   │   └── google_embedding_matcher.py
│   ├── services/
│   │   └── matching_service.py    ← Core API
│   ├── preprocessing/
│   ├── evaluation/
│   └── llm/
├── scripts/
│   ├── create_sample_data.py
│   ├── download_datasets.py
│   └── index_qdrant.py
├── data/
│   ├── sample_resumes/
│   └── sample_job_descriptions/
└── tests/
```

---

## 🔌 Python API (Direct Use)

```python
from src.services.matching_service import match_resumes, ResumeDocument
from src.preprocessing.skill_extractor import SkillExtractor

# Prepare documents
jd_text = open("job.txt").read()
resumes = [
    ResumeDocument(candidate_id="john", filename="john.txt", text=open("john.txt").read()),
    ResumeDocument(candidate_id="jane", filename="jane.txt", text=open("jane.txt").read()),
]

# Run matching
skill_extractor = SkillExtractor()
results = match_resumes(
    jd_text=jd_text,
    resumes=resumes,
    model_key="e5",  # tfidf, bm25, sbert, e5, bge, google
    weights={"semantic": 0.5, "skills": 0.3, "experience": 0.1, "education": 0.1},
    skill_extractor=skill_extractor,
)

# Print results
for r in results:
    print(f"{r.rank}. {r.candidate_id}: {r.final_score:.2f}/100")
```

---

## 📋 Output Formats

### Table (default)
```
Rank   Candidate    Model    Semantic   Final    Recommendation
1      john         SBERT    85.34      82.50    Strong Match
2      jane         SBERT    72.15      69.80    Potential
```

### Verbose (-v, --verbose)
Chi tiết skills, experience, education, score breakdown

### CSV (--output file.csv)
Exported to file, tất cả columns

### JSON (--json)
Pretty JSON format, dễ parse

---

## ⚙️ Environment Variables

```bash
# .env file hoặc environment variables

# Required for Google Embeddings
GOOGLE_API_KEY="your-key"
# hoặc
GEMINI_API_KEY="your-key"

# Model overrides (optional)
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
E5_MODEL="intfloat/e5-base-v2"
BGE_MODEL="BAAI/bge-base-en-v1.5"
GOOGLE_EMBEDDING_MODEL="models/gemini-embedding-001"

# Qdrant (optional)
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="your-key"
QDRANT_LOCAL_PATH="outputs/qdrant_local"
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'sentence_transformers'` | `pip install -r requirements.txt` |
| Model slow first time | Normal! Downloaded from Hugging Face, then cached |
| GPU not used | Install `torch` with CUDA support |
| Google Embeddings fail | Set `GOOGLE_API_KEY` environment variable |
| PDF extraction empty | Use .txt or .docx instead (no OCR yet) |

---

## 📊 Supported Models

| Model | Speed | Quality | Requirements |
|-------|-------|---------|--------------|
| **TF-IDF** | ⚡⚡⚡ | ⭐⭐ | - |
| **BM25** | ⚡⚡⚡ | ⭐⭐⭐ | - |
| **SBERT** | ⚡⚡ | ⭐⭐⭐⭐ | GPU optional |
| **E5** | ⚡⚡ | ⭐⭐⭐⭐ | GPU optional |
| **BGE** | ⚡ | ⭐⭐⭐⭐⭐ | GPU optional |
| **Google** | ⚡⚡ | ⭐⭐⭐⭐⭐ | API key |

---

## 📖 Full Documentation

See **CLI_GUIDE.md** for detailed guide with:
- Advanced examples
- Batch processing
- API usage
- Performance tips
- Output examples
- And more!

---

**Version:** 1.0  
**Last updated:** 2024  
**Author:** AI Resume Screening Project
