from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DEFAULT_HYBRID_WEIGHTS,
    MODEL_NAMES,
    PROCESSED_DATA_DIR,
    QDRANT_JOB_COLLECTION_NAME,
    QDRANT_JOB_EMBEDDING_MODEL,
    QDRANT_URL,
    SAMPLE_JD_DIR,
    SAMPLE_RESUME_DIR,
)
from src.evaluation.benchmark import run_benchmark
from src.evaluation.visualization import job_score_bar, model_comparison_chart
from src.llm.chatbot_chain import answer_job_match_question
from src.llm.explanation_chain import generate_candidate_explanation
from src.llm.structured_extraction import extract_cv_json, extract_job_description_json
from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.skill_extractor import SkillExtractor
from src.rag.qdrant_job_retriever import QdrantJobRetriever
from src.services.export_service import job_matching_report_markdown, job_results_to_csv_bytes, job_results_to_dataframe
from src.services.matching_service import JobDescriptionDocument, ResumeDocument, match_jobs_to_resume
from src.utils.constants import MODEL_SBERT
from src.utils.file_utils import shorten_text



st.set_page_config(
    page_title="AI Job Match Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "results": [],
        "resume_text": "",
        "resume_filename": "uploaded_resume.txt",
        "job_docs": [],
        "job_docs_signature": [],
        "job_descriptions_text": "",
        "explanations": {},
        "chat_history": [],
        "rag_results": [],
        "sbert_rag_requested": False,
        "structured_extraction": {},

    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def parse_uploaded_text(uploaded_file, parser: DocumentParser) -> str:
    parsed = parser.parse(uploaded_file, filename=uploaded_file.name)
    return parsed.cleaned_text


def job_docs_signature(job_docs: list[JobDescriptionDocument]) -> list[tuple[str, str, int, str, str]]:
    return [(job.job_id, job.filename, len(job.text), job.text[:120], job.text[-120:]) for job in job_docs]


def load_sample_documents() -> tuple[str, str, list[JobDescriptionDocument]]:
    parser = DocumentParser()
    resume_files = sorted(SAMPLE_RESUME_DIR.glob("*.txt"))
    jd_files = sorted(SAMPLE_JD_DIR.glob("*.txt"))
    if resume_files and jd_files:
        resume_path = resume_files[0]
        resume_text = parser.parse_path(resume_path).cleaned_text
        jobs = [
            JobDescriptionDocument(job_id=path.stem, filename=path.name, text=parser.parse_path(path).cleaned_text)
            for path in jd_files
        ]
        return resume_path.name, resume_text, jobs

    resume_text = """Machine learning engineer with 4 years of experience building NLP and analytics products.
Skills: Python, SQL, PyTorch, scikit-learn, pandas, Docker, AWS, REST APIs, and model evaluation.
Projects: Built a resume-job matching prototype with sentence embeddings, hybrid scoring, and explainable skill analysis.
Education: Bachelor of Science in Computer Science."""
    jobs = [
        JobDescriptionDocument(
            job_id="machine_learning_engineer",
            filename="machine_learning_engineer.txt",
            text=(
                "Machine Learning Engineer required with Python, SQL, PyTorch, NLP, Docker, AWS, "
                "model evaluation, API deployment, and 3 years of experience."
            ),
        ),
        JobDescriptionDocument(
            job_id="frontend_developer",
            filename="frontend_developer.txt",
            text=(
                "Frontend Developer needed with React, TypeScript, CSS, design systems, browser testing, "
                "and UI accessibility experience."
            ),
        ),
        JobDescriptionDocument(
            job_id="data_analyst",
            filename="data_analyst.txt",
            text=(
                "Data Analyst role requiring SQL, Excel, dashboard reporting, stakeholder communication, "
                "statistics, and business KPI analysis."
            ),
        ),
    ]
    return "sample_resume.txt", resume_text, jobs


# def display_extracted_keywords() -> None:
#     """Display extracted keywords from JD and resumes for testing."""
#     st.markdown("### 🔍 Extracted Keywords (Skills)")
    
#     jd_text = st.session_state.get("jd_text", "")
#     resume_docs = st.session_state.get("resume_docs", [])
    
#     if not jd_text and not resume_docs:
#         st.info("Upload JD and resumes to see extracted keywords")
#         return
    
#     try:
#         extractor = SkillExtractor()
        
#         # Extract JD keywords
#         col1, col2 = st.columns(2)
#         with col1:
#             if jd_text:
#                 jd_skills = extractor.extract(jd_text)
#                 st.markdown("**Job Description Keywords**")
#                 if jd_skills:
#                     st.success(f"Found {len(jd_skills)} skills")
#                     # Display in columns for better layout
#                     skills_text = ", ".join(jd_skills)
#                     st.text_area("JD Skills:", value=skills_text, height=100, disabled=True)
#                 else:
#                     st.warning("No keywords found")
        
#         # Extract Resume keywords
#         with col2:
#             if resume_docs:
#                 st.markdown("**Resume Keywords**")
#                 for doc in resume_docs:
#                     resume_skills = extractor.extract(doc.text)
#                     st.markdown(f"*{doc.filename}*")
#                     if resume_skills:
#                         st.success(f"Found {len(resume_skills)} skills")
#                         skills_text = ", ".join(resume_skills)
#                         st.text_area(f"Skills: {doc.filename}", value=skills_text, height=80, disabled=True)
#                     else:
#                         st.warning("No keywords found")
        
#         # Show keyword comparison
#         if jd_text and resume_docs:
#             st.markdown("---")
#             st.markdown("**Keyword Comparison**")
#             jd_skills = extractor.extract(jd_text)
            
#             for doc in resume_docs:
#                 resume_skills = extractor.extract(doc.text)
#                 matched = sorted(set(jd_skills) & set(resume_skills))
#                 missing = sorted(set(jd_skills) - set(resume_skills))
                
#                 col_match, col_missing = st.columns(2)
#                 with col_match:
#                     st.markdown(f"✅ **Matched** ({doc.filename})")
#                     if matched:
#                         st.write(", ".join(matched))
#                     else:
#                         st.write("No matches")
                
#                 with col_missing:
#                     st.markdown(f"❌ **Missing** ({doc.filename})")
#                     if missing:
#                         st.write(", ".join(missing))
#                     else:
#                         st.write("All keywords present!")
    
#     except Exception as e:
#         st.error(f"Error extracting keywords: {e}")

def display_structured_extraction() -> None:
    st.markdown("### Structured JSON Extraction")
    resume_text = st.session_state.get("resume_text", "")
    job_docs = st.session_state.get("job_docs", [])

    if not resume_text or not job_docs:
        st.info("Upload a resume/CV and job descriptions to extract structured JSON.")
        return

    use_llm = st.toggle(
        "Use Gemini LLM extraction",
        value=True,
        help="Uses GOOGLE_API_KEY or GEMINI_API_KEY when configured. If unavailable, the app returns the same JSON schema with deterministic extraction.",
    )

    if st.button("Extract structured JSON"):
        try:
            with st.spinner("Extracting structured JSON..."):
                extractor = SkillExtractor()
                rows = []
                for job in job_docs:
                    jd_extraction = extract_job_description_json(job.text, use_llm=use_llm, skill_extractor=extractor)
                    resume_extraction = extract_cv_json(
                        resume_text,
                        jd_extraction,
                        use_llm=use_llm,
                        skill_extractor=extractor,
                    )
                    rows.append(
                        {
                            "job_id": job.job_id,
                            "filename": job.filename,
                            "job_description": jd_extraction,
                            "resume": resume_extraction,
                        }
                    )
                st.session_state["structured_extraction"] = {
                    "resume": {
                        "candidate_id": "uploaded_resume",
                        "filename": st.session_state.get("resume_filename", "uploaded_resume.txt"),
                    },
                    "job_descriptions": rows,
                }
        except Exception as exc:
            st.error(str(exc))

    extraction = st.session_state.get("structured_extraction", {})
    if extraction:
        json_text = json.dumps(extraction, ensure_ascii=False, indent=2)
        st.download_button(
            "Download extraction JSON",
            data=json_text,
            file_name="structured_extraction.json",
            mime="application/json",
        )
        st.json(extraction)


def display_sbert_qdrant_jobs(results) -> None:
    st.subheader("SBERT Qdrant Top-K Jobs Related to Resume")
    st.caption(
        "Flow: clean the full resume text, extract CV fields, embed one full query, "
        "retrieve top-K jobs from Qdrant, and display the returned payload metadata. "
        "This does not modify the SBERT semantic score or final ranking score."
    )

    evidence = None
    for result in results:
        result_evidence = getattr(result, "rag_evidence", None)
        if result_evidence is not None:
            evidence = result_evidence
            break
    if evidence is None:
        st.warning("SBERT RAG evidence could not be generated for this candidate.")
        return
    if not evidence:
        st.info("No Qdrant jobs were retrieved for this resume.")
        return
    if not any(isinstance(item, dict) and item.get("job_id") for item in evidence):
        st.warning(
            "These saved results do not contain Qdrant job metadata. "
            "Click 'Find best matching job' again to refresh the top-K Qdrant jobs."
        )
        return

    candidate_name = evidence[0].get("candidate_name") or "uploaded resume"
    with st.expander(f"SBERT RAG Evidence - {candidate_name}", expanded=True):
        rows = [
            {
                "rank": item.get("rank"),
                "score": round(float(item.get("score", 0.0)), 4),
                "job_id": item.get("job_id"),
                "payload_chunk_id": item.get("chunk_id"),
                "payload_chunk_index": item.get("chunk_index"),
                "job_title": item.get("job_title"),
                "category": item.get("category") or item.get("job_category"),
                "required_skills": shorten_text(str(item.get("required_skills") or ""), 180),
                "job_description_preview": shorten_text(str(item.get("job_description") or ""), 220),
                "qdrant_text_preview": shorten_text(str(item.get("text") or item.get("chunk_text") or ""), 220),
            }
            for item in evidence
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for item in evidence:
            qdrant_text = item.get("text") or item.get("chunk_text") or ""
            st.markdown("---")
            st.markdown(
                f"**Rank {item.get('rank')}** | "
                f"similarity score: `{float(item.get('score', 0.0)):.4f}` | "
                f"job id: `{item.get('job_id') or 'Unknown'}` | "
                f"category: `{item.get('category') or item.get('job_category') or 'Unknown'}`"
            )
            st.markdown(f"**Job title:** {item.get('job_title') or 'Unknown'}")
            st.markdown(
                f"**Qdrant payload IDs:** chunk_id `{item.get('chunk_id') or 'Unknown'}`, "
                f"chunk_index `{item.get('chunk_index')}`"
            )
            st.markdown(f"**Required skills:** {item.get('required_skills') or 'Unknown'}")
            st.markdown("**Full job description metadata**")
            st.write(item.get("job_description") or "Not available")
            st.markdown("**Qdrant text payload**")
            st.write(qdrant_text)
            st.markdown("**Full Qdrant payload**")
            st.json(item.get("payload", {}))


def make_weight_controls() -> dict[str, float]:
    st.caption("Hybrid scoring weights")
    semantic = st.slider("Semantic similarity", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["semantic"], 0.05)
    skills = st.slider("Required skills", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["skills"], 0.05)
    experience = st.slider("Experience", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["experience"], 0.05)
    education = st.slider("Education/certification", 0.0, 1.0, DEFAULT_HYBRID_WEIGHTS["education"], 0.05)
    return {
        "semantic": semantic,
        "skills": skills,
        "experience": experience,
        "education": education,
    }


def home_page() -> None:
    st.title("AI-Powered Job Match Assistant")
    st.write(
        "This bachelor-thesis prototype ranks multiple job descriptions against one uploaded resume/CV using "
        "baseline IR models, semantic embedding models, and an interpretable hybrid score."
    )
    st.warning(
        "The system recommends likely job matches from document evidence. It does not make final employment decisions."
    )

    st.subheader("System Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["One Resume + Many Job Descriptions"] --> B["Text Extraction"]
            B --> C["Cleaning and Skill Extraction"]
            C --> D["Model Scoring"]
            D --> E["Hybrid Score"]
            E --> F["Job Match Ranking"]
            C --> Q["Qdrant Vector Store"]
            Q --> R["RAG Retrieval Context"]
            F --> G["LLM Explanation"]
            R --> G
            F --> H["Evaluation and Export"]
        ```
        """
    )

    st.subheader("Implemented Models")
    st.table(
        pd.DataFrame(
            [
                {"Model": "TF-IDF", "Role": "Keyword baseline", "Runs locally": "Yes"},
                {"Model": "BM25", "Role": "Information retrieval baseline", "Runs locally": "Yes"},
                {"Model": "Word2Vec", "Role": "Static word embedding baseline", "Runs locally": "Yes, after vector download"},
                {"Model": "GloVe", "Role": "Static word embedding baseline", "Runs locally": "Yes, after vector download"},
                {"Model": "SBERT", "Role": "Semantic sentence embedding", "Runs locally": "Yes"},
                {"Model": "E5", "Role": "Retrieval-optimized embedding", "Runs locally": "Yes"},
                {"Model": "BGE", "Role": "Strong open embedding model", "Runs locally": "Yes"},
                {"Model": "Google Embeddings", "Role": "Optional API comparison", "Runs locally": "Requires API key"},
                {"Model": "Qdrant", "Role": "Vector database for RAG retrieval", "Runs locally": "Yes or server"},
            ]
        )
    )


def screening_page() -> None:
    st.title("Job Match Recommendation")
    st.write("Upload one resume/CV, compare it with multiple job descriptions, and rank the jobs by fit.")
    parser = DocumentParser()

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Resume/CV")
        resume_text = st.text_area("Paste resume/CV", value=st.session_state.get("resume_text", ""), height=260)
        resume_file = st.file_uploader("Upload resume/CV file", type=["pdf", "docx", "txt"], key="resume_file")
        if resume_file:
            try:
                resume_text = parse_uploaded_text(resume_file, parser)
                st.session_state["resume_filename"] = resume_file.name
                st.success(f"Loaded {resume_file.name}")
            except Exception as exc:
                st.error(str(exc))
        if st.button("Load sample data"):
            resume_filename, sample_resume_text, sample_jobs = load_sample_documents()
            st.session_state["resume_filename"] = resume_filename
            st.session_state["resume_text"] = sample_resume_text
            st.session_state["job_docs"] = sample_jobs
            st.session_state["job_docs_signature"] = job_docs_signature(sample_jobs)
            st.session_state["job_descriptions_text"] = ""
            st.session_state["structured_extraction"] = {}
            st.session_state["results"] = []
            st.rerun()
        if resume_text != st.session_state.get("resume_text", ""):
            st.session_state["structured_extraction"] = {}
            st.session_state["results"] = []
        st.session_state["resume_text"] = resume_text

    with right:
        st.subheader("Job Descriptions")
        pasted_jobs = st.text_area(
            "Paste job descriptions",
            value=st.session_state.get("job_descriptions_text", ""),
            height=180,
            help="Separate multiple pasted job descriptions with a line containing only ---.",
        )
        st.session_state["job_descriptions_text"] = pasted_jobs
        uploaded_jobs = st.file_uploader(
            "Upload job description files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="job_description_files",
        )
        parsed_jobs: list[JobDescriptionDocument] = []
        pasted_parts = [part.strip() for part in pasted_jobs.split("\n---\n") if part.strip()]
        for idx, text in enumerate(pasted_parts, start=1):
            parsed_jobs.append(JobDescriptionDocument(job_id=f"pasted_job_{idx}", filename=f"pasted_job_{idx}.txt", text=text))
        if uploaded_jobs:
            for idx, uploaded in enumerate(uploaded_jobs, start=1):
                try:
                    text = parse_uploaded_text(uploaded, parser)
                    parsed_jobs.append(JobDescriptionDocument(job_id=f"job_{idx}", filename=uploaded.name, text=text))
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")
        if parsed_jobs:
            signature = job_docs_signature(parsed_jobs)
            if signature != st.session_state.get("job_docs_signature", []):
                st.session_state["structured_extraction"] = {}
                st.session_state["results"] = []
            st.session_state["job_docs"] = parsed_jobs
            st.session_state["job_docs_signature"] = signature

        job_docs = st.session_state.get("job_docs", [])
        st.write(f"Loaded job descriptions: {len(job_docs)}")
        if st.button("Clear job descriptions"):
            st.session_state["job_docs"] = []
            st.session_state["job_docs_signature"] = []
            st.session_state["job_descriptions_text"] = ""
            st.session_state["structured_extraction"] = {}
            st.session_state["results"] = []
            st.rerun()

    with st.expander("Text extraction preview", expanded=False):
        st.markdown("**Resume/CV preview**")
        st.text(shorten_text(st.session_state.get("resume_text", ""), 1500))
        for job in st.session_state.get("job_docs", []):
            st.markdown(f"**{job.filename}**")
            st.text(shorten_text(job.text, 1000))

    # # Display extracted keywords section
    # display_extracted_keywords()
    display_structured_extraction()

    st.subheader("Model and Scoring")
    model_label = st.selectbox(
        "Matching model",
        options=list(MODEL_NAMES.keys()),
        format_func=lambda key: MODEL_NAMES[key],
        index=0,
    )
    use_hybrid = st.toggle("Use hybrid score", value=True)
    use_structured_score = st.toggle(
        "Use structured JSON for hybrid score",
        value=True,
        disabled=not use_hybrid,
        help="Calculates skill, experience, and education scores from the structured JD/CV JSON instead of raw cleaned text.",
    )
    use_llm_scoring_extraction = st.toggle(
        "Use Gemini extraction during scoring",
        value=True,
        disabled=not use_hybrid or not use_structured_score,
        help="If no extracted JSON is already available, scoring can call Gemini. Without an API key it falls back to deterministic JSON extraction.",
    )
    weights = make_weight_controls()
    show_sbert_rag = False
    sbert_rag_top_k = 5
    if model_label == MODEL_SBERT:
        show_sbert_rag = st.checkbox("Show Qdrant top-K jobs related to resume", value=False)
        if show_sbert_rag:
            sbert_rag_top_k = st.slider("Top-K Qdrant jobs", min_value=1, max_value=10, value=5, step=1)

    if st.button("Find best matching job", type="primary"):
        try:
            resume = ResumeDocument(
                candidate_id="uploaded_resume",
                filename=st.session_state.get("resume_filename", "uploaded_resume.txt"),
                text=st.session_state.get("resume_text", ""),
            )
            with st.spinner("Scoring job descriptions..."):
                st.session_state["results"] = match_jobs_to_resume(
                    resume=resume,
                    job_descriptions=st.session_state.get("job_docs", []),
                    model_key=model_label,
                    weights=weights,
                    use_hybrid_score=use_hybrid,
                    use_structured_extraction_score=use_structured_score,
                    use_llm_extraction=use_llm_scoring_extraction,
                    structured_extraction=st.session_state.get("structured_extraction") or None,
                    include_sbert_rag_evidence=model_label == MODEL_SBERT and show_sbert_rag,
                    sbert_rag_top_k=int(sbert_rag_top_k),
                )
                st.session_state["sbert_rag_requested"] = model_label == MODEL_SBERT and show_sbert_rag
                st.session_state["explanations"] = {}
                st.session_state["chat_history"] = []
        except Exception as exc:
            st.error(str(exc))

    results = st.session_state.get("results", [])
    if not results:
        return

    best = results[0]
    st.success(f"Best match: {best.filename} with a {best.final_score:.2f}% final score ({best.recommendation}).")

    st.subheader("Ranked Job Descriptions")
    table = job_results_to_dataframe(results)
    visible_columns = [
        "rank",
        "job_description_file",
        "final_score",
        "semantic_score",
        "recommendation",
        "matched_skills",
        "missing_skills",
        "runtime_seconds",
    ]
    display_table = table[visible_columns].rename(
        columns={
            "job_description_file": "job_description",
            "matched_skills": "matched_resume_skills",
            "missing_skills": "missing_job_skills",
        }
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)
    st.plotly_chart(job_score_bar(results), use_container_width=True)
    if (
        st.session_state.get("sbert_rag_requested")
        and results
        and all(result.model_key == MODEL_SBERT for result in results)
    ):
        display_sbert_qdrant_jobs(results)

    csv_bytes = job_results_to_csv_bytes(results)
    st.download_button("Export ranking CSV", data=csv_bytes, file_name="job_matching_results.csv", mime="text/csv")
    report = job_matching_report_markdown(st.session_state["resume_text"], results)
    st.download_button("Export job matching report", data=report, file_name="job_matching_report.md", mime="text/markdown")

    st.subheader("Job Match Detail Analysis")
    selected = st.selectbox(
        "Job description",
        options=[result.filename for result in results],
        index=0,
    )
    result = next(item for item in results if item.filename == selected)
    selected_job = next((job for job in st.session_state.get("job_docs", []) if job.filename == selected), None)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{result.final_score:.2f}%")
    c2.metric("Semantic score", f"{result.semantic_score:.2f}%")
    c3.metric("Resume years", f"{result.detected_resume_years:g}")
    c4.metric("Job required years", f"{result.required_years:g}")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**Matched resume skills**")
        st.write(", ".join(result.matched_skills) or "None detected")
        st.markdown("**Education/certification evidence**")
        evidence = result.education_signals + result.certification_signals
        st.write(", ".join(evidence) or "No dictionary evidence detected")
    with detail_right:
        st.markdown("**Missing job-required skills**")
        st.write(", ".join(result.missing_skills) or "None detected")
        st.markdown("**Match label**")
        st.write(result.recommendation)

    if selected_job:
        with st.expander("Selected job description preview", expanded=False):
            st.text(shorten_text(selected_job.text, 1800))

    if result.structured_extraction:
        with st.expander("Structured extraction used for scoring", expanded=False):
            st.json(result.structured_extraction)

    if st.button("Generate explanation"):
        job_text = selected_job.text if selected_job else ""
        explanation = generate_candidate_explanation(result, job_text, result.cleaned_resume_text, use_llm=True)
        st.session_state["explanations"][result.filename] = explanation
    if result.filename in st.session_state["explanations"]:
        st.markdown(st.session_state["explanations"][result.filename])

    st.subheader("Job Match Chatbot")
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask about top jobs, missing skills, or why a job ranked highly")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        answer = answer_job_match_question(question, results, st.session_state["resume_text"], use_llm=True)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()


def evaluation_page() -> None:
    st.title("Evaluation Dashboard")
    pairs_path = PROCESSED_DATA_DIR / "test_pairs.csv"
    if not pairs_path.exists():
        st.info("Prepare dataset pairs first. The dashboard expects data/processed/test_pairs.csv.")
        st.code(
            "python -m src.data.prepare_pairs --primary-pairs data/raw/<ranking_dataset_file>.csv",
            language="bash",
        )
        return

    models = st.multiselect("Models", options=list(MODEL_NAMES.keys()), default=["tfidf", "bm25"], format_func=lambda key: MODEL_NAMES[key])
    limit_jobs = st.number_input("Job limit for quick run", min_value=1, max_value=1000, value=20)
    if st.button("Run evaluation"):
        if not models:
            st.warning("Select at least one model.")
            return
        try:
            with st.spinner("Running evaluation..."):
                comparison = run_benchmark(pairs_path, models, limit_jobs=int(limit_jobs))
            st.session_state["comparison"] = comparison
        except Exception as exc:
            st.error(str(exc))

    comparison_path = ROOT / "outputs" / "evaluation_results" / "model_comparison.csv"
    comparison = st.session_state.get("comparison")
    if comparison is None and comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

    if comparison is not None:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        metric_columns = [col for col in comparison.columns if "@" in col or col in ["mrr", "map"]]
        if metric_columns:
            metric = st.selectbox("Metric", metric_columns, index=0)
            st.plotly_chart(model_comparison_chart(comparison, metric), use_container_width=True)


def rag_page() -> None:
    st.title("Qdrant Resume-to-Job Metadata Retrieval")
    st.write(
        "This page cleans one full resume/CV, extracts CV fields, embeds one full query, retrieves top-K related jobs "
        "from Qdrant, and displays the returned payload metadata."
    )

    c1, c2, c3 = st.columns(3)
    resume_text = st.session_state.get("resume_text", "")
    c1.metric("Loaded resume", "Yes" if resume_text.strip() else "No")
    c2.metric("Qdrant collection", QDRANT_JOB_COLLECTION_NAME)
    c3.metric("Embedding model", QDRANT_JOB_EMBEDDING_MODEL.split("/")[-1])

    if QDRANT_URL:
        st.caption(f"Qdrant server: {QDRANT_URL}")

    if not resume_text.strip():
        st.info("Load or upload a resume/CV on the Job Matching page first, or use the sample data button below.")
        if st.button("Load sample documents for RAG"):
            resume_filename, sample_resume_text, job_docs = load_sample_documents()
            st.session_state["resume_filename"] = resume_filename
            st.session_state["resume_text"] = sample_resume_text
            st.session_state["job_docs"] = job_docs
            st.session_state["job_docs_signature"] = job_docs_signature(job_docs)
            st.rerun()

    query_source = st.radio("Query source", ["Loaded resume/CV", "Paste resume text"], horizontal=True)
    if query_source == "Loaded resume/CV" and resume_text.strip():
        query_text = resume_text
        with st.expander("Full query resume/CV", expanded=False):
            st.text(shorten_text(query_text, 2500))
    else:
        query_text = st.text_area(
            "Full resume/CV query",
            value=resume_text
            or (
                "Human resources generalist with experience in employee relations, talent acquisition, onboarding, "
                "performance management, HR compliance, policy administration, and HRIS reporting."
            ),
            height=180,
        )

    use_llm = st.toggle(
        "Use Gemini CV field extraction for Qdrant query",
        value=True,
        help="If Gemini is unavailable, the existing deterministic extractor is used as fallback.",
    )
    limit = st.slider("Top-K Qdrant jobs", min_value=1, max_value=10, value=5)

    if st.button("Retrieve top-K Qdrant jobs", type="primary"):
        try:
            with st.spinner("Embedding full resume/CV and retrieving jobs from Qdrant..."):
                retriever = QdrantJobRetriever()
                st.session_state["rag_results"] = retriever.retrieve_resume_job_evidence(
                    query_text,
                    top_k=limit,
                    candidate_name=st.session_state.get("resume_filename", "uploaded_resume.txt"),
                    resume_id="uploaded_resume",
                    use_llm_extraction=use_llm,
                )
        except Exception as exc:
            st.error(str(exc))

    results = st.session_state.get("rag_results", [])
    if not results:
        return

    rows = [
        {
            "score": round(result.score, 4),
            "job_id": result.job_id,
            "chunk_id": result.chunk_id,
            "chunk_index": result.chunk_index,
            "job_title": result.job_title,
            "category": result.category,
            "required_skills": shorten_text(result.required_skills or "", 220),
            "job_description_preview": shorten_text(result.job_description or "", 260),
            "qdrant_text_preview": shorten_text(result.text or result.chunk_text or "", 260),
        }
        for result in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for result in results:
        with st.expander(f"Full Qdrant payload - rank {result.rank}: {result.job_title or result.job_id}", expanded=False):
            st.json(result.payload)


def dataset_page() -> None:
    st.title("Dataset and Thesis Notes")
    st.write("Primary evaluation should use a paired resume-JD dataset with a match score when available.")
    st.table(
        pd.DataFrame(
            [
                {
                    "Dataset": "Resume Data for Ranking",
                    "Kaggle slug": "thejohnwick001/resume-data-for-ranking",
                    "Use": "Primary paired ranking evaluation with matched_score.",
                },
                {
                    "Dataset": "recruitment dataset",
                    "Kaggle slug": "surendra365/recruitement-dataset",
                    "Use": "Alternative labeled synthetic resume-JD pairs.",
                },
                {
                    "Dataset": "Job Descriptions 2025",
                    "Kaggle slug": "adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles",
                    "Use": "Job descriptions for weak pairing and demo.",
                },
                {
                    "Dataset": "Resume Dataset",
                    "Kaggle slug": "snehaanbhawal/resume-dataset",
                    "Use": "Resume category dataset for weak labels and extraction tests.",
                },
            ]
        )
    )
    st.code(
        "kaggle datasets download -d thejohnwick001/resume-data-for-ranking -p data/raw --unzip\n"
        "python -m src.data.prepare_pairs --primary-pairs data/raw/<file>.csv\n"
        "python scripts/index_qdrant.py\n"
        "python -m src.evaluation.benchmark --pairs data/processed/test_pairs.csv --models tfidf,bm25,word2vec,glove,sbert",
        language="bash",
    )


def main() -> None:
    init_state()
    page = st.sidebar.radio("Navigation", ["Home", "Job Matching", "Evaluation", "Qdrant Job Retrieval", "Dataset & Thesis"])
    if page == "Home":
        home_page()
    elif page == "Job Matching":
        screening_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Qdrant Job Retrieval":
        rag_page()
    else:
        dataset_page()


if __name__ == "__main__":
    main()
