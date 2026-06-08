# 🚀 Terminal/CLI Guide - Chạy không cần Streamlit

## 1️⃣ Quick Start

### Sử dụng Sample Data (Mẫu sẵn)
```bash
# Chạy với model SBERT (mặc định)
python cli_screening.py

# Chạy với model khác
python cli_screening.py --model e5
python cli_screening.py --model bge
python cli_screening.py --model bm25
python cli_screening.py --model tfidf
python cli_screening.py --model word2vec
python cli_screening.py --model glove
```

### Output mẫu:
```
📄 Loading documents...
   Using sample JD: machine_learning_engineer.txt
   Using 3 sample resumes
✅ Loaded 1 JD and 3 resume(s)

🔍 Screening with SBERT...
====================================================================================================
Rank   Candidate                      Model                Semantic     Final        Recommendation   
====================================================================================================
1      candidate_alex_python_ml       SBERT                85.34        82.50        Strong Match     
2      candidate_chris_data_analyst   SBERT                72.15        69.80        Potential       
3      candidate_brianna_frontend     SBERT                45.20        48.90        Below Threshold  
====================================================================================================

✨ Done! Screening completed in 0.1234s total
```

---

## 2️⃣ Sử dụng File của Riêng Bạn

### Đối số chính:
```bash
# Chỉ định JD
python cli_screening.py --jd path/to/job.txt

# Chỉ định một hoặc nhiều resume
python cli_screening.py --resumes path/to/resume1.txt path/to/resume2.txt path/to/resume3.txt

# Cả hai
python cli_screening.py \
  --jd jobs/machine_learning_engineer.txt \
  --resumes resumes/john.txt resumes/jane.txt resumes/bob.txt \
  --model e5
```

### Format file hỗ trợ:
- ✅ `.txt` - Plain text
- ✅ `.pdf` - PDF files
- ✅ `.docx` - Microsoft Word
- ✅ `.doc` - Older Word documents

---

## 3️⃣ Chọn Model

### Available Models:
```bash
# IR Baselines
python cli_screening.py --model tfidf    # TF-IDF + Cosine Similarity (nhanh nhất)
python cli_screening.py --model bm25     # BM25 (keyword-based, mạnh)
python cli_screening.py --model word2vec # Word2Vec static embedding baseline
python cli_screening.py --model glove    # GloVe static embedding baseline

# Semantic Embeddings
python cli_screening.py --model sbert    # SBERT (mẫu)
python cli_screening.py --model e5       # E5 (retrieval-optimized)
python cli_screening.py --model bge      # BGE (strong general model)
python cli_screening.py --model google   # Google Embeddings (requires API key)
```

### So sánh nhanh:
| Model | Tốc độ | Chất lượng | Requirement |
|-------|--------|-----------|-------------|
| TF-IDF | ⚡⚡⚡ | ⭐⭐ | - |
| BM25 | ⚡⚡⚡ | ⭐⭐⭐ | - |
| Word2Vec | ⚡⚡ | ⭐⭐⭐ | pretrained vectors |
| GloVe | ⚡⚡ | ⭐⭐⭐ | pretrained vectors |
| SBERT | ⚡⚡ | ⭐⭐⭐⭐ | GPU optional |
| E5 | ⚡⚡ | ⭐⭐⭐⭐ | GPU optional |
| BGE | ⚡ | ⭐⭐⭐⭐⭐ | GPU optional |
| Google | ⚡⚡ | ⭐⭐⭐⭐⭐ | API key required |

---

## 4️⃣ Tùy chỉnh Hybrid Scoring

### Weights mặc định:
```
Semantic: 50%
Skills: 30%
Experience: 10%
Education: 10%
```

### Thay đổi weights:
```bash
# Nhấn mạnh semantic hơn
python cli_screening.py --semantic 0.7 --skills 0.2 --experience 0.05 --education 0.05

# Nhấn mạnh skill matching
python cli_screening.py --semantic 0.3 --skills 0.5 --experience 0.1 --education 0.1

# Chỉ dùng semantic similarity (no hybrid)
python cli_screening.py --semantic 1.0 --skills 0 --experience 0 --education 0
```

**Lưu ý:** Weights phải cộng lại = 1.0

---

## 5️⃣ Output Options

### Bảng đơn giản (mặc định):
```bash
python cli_screening.py
```

### Chi tiết từng candidate:
```bash
python cli_screening.py --verbose
# hoặc
python cli_screening.py -v
```

Output sẽ bao gồm:
- ✅ Matched skills
- ❌ Missing skills
- 📊 Years of experience detected vs required
- 🎓 Education/Certification signals
- 📈 Score breakdown (semantic, skills, experience, education)

### Xuất CSV:
```bash
python cli_screening.py --output results.csv
```

File CSV sẽ có tất cả columns:
- rank, candidate_id, filename, model_name
- semantic_score, final_score, recommendation
- matched_skills, missing_skills, resume_skills, jd_skills
- detected_resume_years, required_years
- education_signals, certification_signals, runtime_seconds

### Xuất JSON:
```bash
python cli_screening.py --json

# Lưu JSON vào file
python cli_screening.py --json > results.json
```

### Kết hợp options:
```bash
python cli_screening.py \
  --model e5 \
  --verbose \
  --output results.csv \
  --jd jobs/senior_dev.txt \
  --resumes resumes/*.txt
```

---

## 6️⃣ Ví dụ Thực tế

### Example 1: Tìm senior developer tốt nhất
```bash
python cli_screening.py \
  --jd data/sample_job_descriptions/machine_learning_engineer.txt \
  --resumes data/sample_resumes/*.txt \
  --model bge \
  --semantic 0.6 \
  --skills 0.3 \
  --experience 0.07 \
  --education 0.03 \
  --verbose
```

### Example 2: Nhanh chóng qua lô hàng lớn
```bash
python cli_screening.py \
  --jd jobs/position.txt \
  --resumes candidates/*.txt \
  --model tfidf \
  --output quick_screening.csv
```

### Example 3: Chi tiết và lưu đầy đủ
```bash
python cli_screening.py \
  --jd jobs/frontend_dev.txt \
  --resumes resumes/john.txt resumes/jane.txt \
  --model e5 \
  --verbose \
  --json > detailed_results.json
```

### Example 4: Chỉ semantic similarity (không hybrid)
```bash
python cli_screening.py \
  --model bge \
  --semantic 1.0 \
  --skills 0 \
  --experience 0 \
  --education 0 \
  --verbose
```

---

## 7️⃣ Advanced - Python API

### Sử dụng trực tiếp trong code Python:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from src.services.matching_service import match_resumes, ResumeDocument
from src.preprocessing.skill_extractor import SkillExtractor

# Tải tài liệu
jd_text = open("jobs/machine_learning_engineer.txt").read()
resumes = [
    ResumeDocument(
        candidate_id="john",
        filename="john.txt",
        text=open("resumes/john.txt").read(),
    ),
    ResumeDocument(
        candidate_id="jane",
        filename="jane.txt",
        text=open("resumes/jane.txt").read(),
    ),
]

# Chạy matching
skill_extractor = SkillExtractor()
results = match_resumes(
    jd_text=jd_text,
    resumes=resumes,
    model_key="e5",  # hoặc "sbert", "bge", "word2vec", "glove", "tfidf", "bm25", "google"
    weights={
        "semantic": 0.5,
        "skills": 0.3,
        "experience": 0.1,
        "education": 0.1,
    },
    use_hybrid_score=True,
    skill_extractor=skill_extractor,
)

# Xử lý kết quả
for result in results:
    print(f"{result.rank}. {result.candidate_id}: {result.final_score:.2f}/100")
    print(f"   Matched skills: {', '.join(result.matched_skills)}")
    print(f"   Missing skills: {', '.join(result.missing_skills)}")
```

### Sử dụng từng model riêng lẻ:

```python
from src.models.sbert_matcher import SBERTMatcher
from src.models.e5_matcher import E5Matcher
from src.models.bge_matcher import BGEMatcher
from src.models.tfidf_matcher import TfidfMatcher
from src.models.bm25_matcher import BM25Matcher
from src.models.word2vec_matcher import Word2VecMatcher
from src.models.glove_matcher import GloveMatcher

jd = "Software Engineer with Python, SQL, AWS experience"
resumes = [
    "Senior Python dev, 5 years AWS, SQL expert",
    "Frontend developer, JavaScript specialist",
]

# SBERT
matcher = SBERTMatcher()
scores = matcher.score(jd, resumes)
print(f"SBERT: {scores}")  # [0.87, 0.34]

# E5
matcher = E5Matcher()
scores = matcher.score(jd, resumes)
print(f"E5: {scores}")  # [0.91, 0.31]

# TF-IDF (nhanh nhất)
matcher = TfidfMatcher()
scores = matcher.score(jd, resumes)
print(f"TF-IDF: {scores}")  # [0.78, 0.42]

# Word2Vec / GloVe static embeddings
matcher = Word2VecMatcher()
scores = matcher.score(jd, resumes)
print(f"Word2Vec: {scores}")  # [0.81, 0.39]

matcher = GloveMatcher()
scores = matcher.score(jd, resumes)
print(f"GloVe: {scores}")  # [0.79, 0.38]
```

---

## 8️⃣ Troubleshooting

### "No module named 'sentence_transformers'"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Model download rất chậm lần đầu
- Bình thường! Models được tải từ Hugging Face một lần rồi cache
- SBERT: ~100MB
- E5: ~140MB
- BGE: ~400MB

### Google Embeddings không hoạt động
```bash
# Set API key
export GOOGLE_API_KEY="your-key"
# hoặc Windows:
$env:GOOGLE_API_KEY="your-key"

python cli_screening.py --model google
```

### PDF không được đọc (trống)
- File PDF có thể là hình ảnh được scan
- Hiện tại chưa hỗ trợ OCR, sử dụng `.txt` hoặc `.docx` thay thế

### Lỗi "Unicode decode error"
- File text cần là UTF-8 encoding
- Đổi encoding sang UTF-8 trước

---

## 9️⃣ Performance Tips

### Để tốc độ tối đa:
```bash
# TF-IDF nhanh nhất
python cli_screening.py --model tfidf

# Hoặc BM25
python cli_screening.py --model bm25
```

### Để chất lượng tối đa:
```bash
# BGE tốt nhất nhưng chậm hơn
python cli_screening.py --model bge

# Hoặc E5 (tốt và nhanh hơn BGE)
python cli_screening.py --model e5
```

### GPU acceleration (nếu có GPU):
```bash
# Tự động sử dụng GPU nếu có PyTorch CUDA
# Không cần thay đổi gì!
python cli_screening.py --model bge
```

---

## 🔟 Batch Processing

### Xử lý nhiều JD:
```bash
#!/bin/bash
# process_all_positions.sh

for jd_file in jobs/*.txt; do
    output="${jd_file%.txt}_results.csv"
    echo "Processing $jd_file..."
    python cli_screening.py \
        --jd "$jd_file" \
        --resumes candidates/*.txt \
        --model e5 \
        --output "$output"
done

echo "All done!"
```

### Windows (PowerShell):
```powershell
Get-ChildItem jobs/*.txt | ForEach-Object {
    $output = $_.BaseName + "_results.csv"
    Write-Host "Processing $_..."
    python cli_screening.py `
        --jd $_.FullName `
        --resumes candidates/*.txt `
        --model e5 `
        --output $output
}
Write-Host "All done!"
```

---

## 📋 CLI Help

```bash
python cli_screening.py --help
```

Hiển thị tất cả available arguments và examples.

---

## 🎯 Next Steps

1. **Benchmark:** Chạy evaluation để so sánh models
   ```bash
   python -m src.evaluation.benchmark --pairs data/processed/test_pairs.csv --models tfidf,bm25,word2vec,glove,e5,bge
   ```

2. **RAG/Vector Store:** Index documents cho semantic search
   ```bash
   python scripts/index_qdrant.py --jd data/sample_job_descriptions/*.txt --resumes data/sample_resumes/*.txt
   ```

3. **LLM Explanations:** Generate explanations (cần API key)
   ```python
   from src.llm.explanation_chain import generate_candidate_explanation
   explanation = generate_candidate_explanation(result)
   print(explanation)
   ```

---

## 📊 Output Examples

### Simple table:
```
Rank   Candidate                      Model                Semantic     Final        Recommendation   
1      candidate_alex_python_ml       SBERT                85.34        82.50        Strong Match     
2      candidate_chris_data_analyst   SBERT                72.15        69.80        Potential       
```

### Verbose output:
```
📋 CANDIDATE_ALEX_PYTHON_ML
   Rank: 1
   Final Score: 82.50/100
   Recommendation: Strong Match
   Runtime: 0.1234s

   Skill Analysis:
      Matched Skills: Python, SQL, PyTorch, Docker, AWS
      Missing Skills: scikit-learn
      Resume has 10 skills
      JD requires 8 skills

   Experience:
      Detected: 4.0 years
      Required: 3.0 years

   Education/Certification:
      Signals: Bachelor in Computer Science
      Certifications: AWS Certified Cloud Practitioner

   Score Breakdown:
      Semantic Similarity: 85.34%
      Skill Match: 87.50%
      Experience Match: 100.00%
      Education Match: 50.00%
```

### CSV output:
```csv
rank,candidate_id,filename,model_name,semantic_score,final_score,recommendation,matched_skills,missing_skills,runtime_seconds
1,candidate_alex,candidate_alex.txt,SBERT,85.34,82.50,Strong Match,"Python, SQL, PyTorch, Docker, AWS",scikit-learn,0.1234
2,candidate_chris,candidate_chris.txt,SBERT,72.15,69.80,Potential,"Python, SQL, Tableau",PyTorch,0.1156
```

### JSON output:
```json
[
  {
    "rank": 1,
    "candidate_id": "candidate_alex",
    "filename": "candidate_alex.txt",
    "model_name": "SBERT",
    "semantic_score": 85.34,
    "final_score": 82.50,
    "recommendation": "Strong Match",
    "matched_skills": ["Python", "SQL", "PyTorch", "Docker", "AWS"],
    "missing_skills": ["scikit-learn"],
    "detected_years": 4.0,
    "required_years": 3.0,
    "runtime_seconds": 0.1234
  }
]
```

---

**Happy screening! 🎉**
