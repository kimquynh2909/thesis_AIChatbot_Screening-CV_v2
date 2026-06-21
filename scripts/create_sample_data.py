# from __future__ import annotations

# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from config.settings import SAMPLE_JD_DIR, SAMPLE_RESUME_DIR
# from src.utils.file_utils import write_text_file


# SAMPLE_JD = """Machine Learning Engineer

# We are hiring a Machine Learning Engineer with 3+ years of experience building Python-based machine learning systems.
# Required skills include Python, SQL, scikit-learn, PyTorch, natural language processing, LangChain, REST API development, Docker, AWS, and data visualization.
# The candidate should hold a bachelor degree in computer science, data science, or a related field.
# Responsibilities include building ML pipelines, evaluating models, deploying APIs, collaborating with product teams, and documenting model limitations.
# AWS certification is preferred.
# """


# RESUMES = {
#     "candidate_alex_python_ml.txt": """Alex Nguyen
# Machine Learning Engineer

# Summary: Machine learning engineer with 4 years of experience building NLP products and model evaluation pipelines.
# Skills: Python, SQL, scikit-learn, PyTorch, pandas, NumPy, natural language processing, LangChain, FastAPI, Docker, AWS, Tableau, data visualization.
# Experience: Built transformer-based text classification services and deployed REST API endpoints with Docker on AWS. Designed model monitoring dashboards and collaborated with product managers.
# Education: Bachelor in Computer Science.
# Certification: AWS Certified Cloud Practitioner.
# """,
#     "candidate_brianna_frontend.txt": """Brianna Tran
# Frontend Developer

# Summary: Frontend developer with 5 years of experience building web applications.
# Skills: JavaScript, TypeScript, React, HTML, CSS, Redux, responsive design, stakeholder management.
# Experience: Built customer dashboards and reusable UI components. Worked in agile teams and mentored junior developers.
# Education: Bachelor in Information Technology.
# """,
#     "candidate_chris_data_analyst.txt": """Chris Le
# Data Analyst

# Summary: Data analyst with 2 years of experience in reporting and analytics.
# Skills: SQL, Python, Excel, Power BI, Tableau, data visualization, statistics, dashboard, KPI analysis.
# Experience: Created business intelligence reports and cleaned datasets with pandas. Supported quarterly planning with KPI dashboards.
# Education: Master of Data Science.
# """,
# }


# def main() -> None:
#     SAMPLE_JD_DIR.mkdir(parents=True, exist_ok=True)
#     SAMPLE_RESUME_DIR.mkdir(parents=True, exist_ok=True)
#     write_text_file(SAMPLE_JD_DIR / "machine_learning_engineer.txt", SAMPLE_JD)
#     for filename, content in RESUMES.items():
#         write_text_file(SAMPLE_RESUME_DIR / filename, content)
#     print(f"Sample data written to {SAMPLE_JD_DIR} and {SAMPLE_RESUME_DIR}")


# if __name__ == "__main__":
#     main()
