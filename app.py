from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_RAG_EMBEDDING_MODEL,
    MODEL_NAMES,
    PROCESSED_DATA_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
    SAMPLE_JD_DIR,
    SAMPLE_RESUME_DIR,
)
from src.evaluation.benchmark import run_benchmark
from src.evaluation.visualization import candidate_score_bar, model_comparison_chart
from src.llm.chatbot_chain import answer_hr_question
from src.llm.explanation_chain import generate_candidate_explanation
from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.skill_extractor import SkillExtractor
from src.services.export_service import results_to_csv_bytes, screening_report_markdown
from src.services.matching_service import ResumeDocument, match_resumes
from src.services.vector_store_service import QdrantRAGStore, build_rag_context, build_rag_documents
from src.utils.file_utils import shorten_text


st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "results": [],
        "jd_text": "",
        "resume_docs": [],
        "explanations": {},
        "chat_history": [],
        "rag_results": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def parse_uploaded_text(uploaded_file, parser: DocumentParser) -> str:
    parsed = parser.parse(uploaded_file, filename=uploaded_file.name)
    return parsed.cleaned_text


def load_sample_documents() -> tuple[str, list[ResumeDocument]]:
    parser = DocumentParser()
    jd_files = sorted(SAMPLE_JD_DIR.glob("*.txt"))
    resume_files = sorted(SAMPLE_RESUME_DIR.glob("*.txt"))
    jd_text = parser.parse_path(jd_files[0]).cleaned_text if jd_files else ""
    resumes = [
        ResumeDocument(candidate_id=path.stem, filename=path.name, text=parser.parse_path(path).cleaned_text)
        for path in resume_files
    ]
    return jd_text, resumes


def display_extracted_keywords() -> None:
    """Display extracted keywords from JD and resumes for testing."""
    st.markdown("### 🔍 Extracted Keywords (Skills)")
    
    jd_text = st.session_state.get("jd_text", "")
    resume_docs = st.session_state.get("resume_docs", [])
    
    if not jd_text and not resume_docs:
        st.info("Upload JD and resumes to see extracted keywords")
        return
    
    try:
        extractor = SkillExtractor()
        
        # Extract JD keywords
        col1, col2 = st.columns(2)
        with col1:
            if jd_text:
                jd_skills = extractor.extract(jd_text)
                st.markdown("**Job Description Keywords**")
                if jd_skills:
                    st.success(f"Found {len(jd_skills)} skills")
                    # Display in columns for better layout
                    skills_text = ", ".join(jd_skills)
                    st.text_area("JD Skills:", value=skills_text, height=100, disabled=True)
                else:
                    st.warning("No keywords found")
        
        # Extract Resume keywords
        with col2:
            if resume_docs:
                st.markdown("**Resume Keywords**")
                for doc in resume_docs:
                    resume_skills = extractor.extract(doc.text)
                    st.markdown(f"*{doc.filename}*")
                    if resume_skills:
                        st.success(f"Found {len(resume_skills)} skills")
                        skills_text = ", ".join(resume_skills)
                        st.text_area(f"Skills: {doc.filename}", value=skills_text, height=80, disabled=True)
                    else:
                        st.warning("No keywords found")
        
        # Show keyword comparison
        if jd_text and resume_docs:
            st.markdown("---")
            st.markdown("**Keyword Comparison**")
            jd_skills = extractor.extract(jd_text)
            
            for doc in resume_docs:
                resume_skills = extractor.extract(doc.text)
                matched = sorted(set(jd_skills) & set(resume_skills))
                missing = sorted(set(jd_skills) - set(resume_skills))
                
                col_match, col_missing = st.columns(2)
                with col_match:
                    st.markdown(f"✅ **Matched** ({doc.filename})")
                    if matched:
                        st.write(", ".join(matched))
                    else:
                        st.write("No matches")
                
                with col_missing:
                    st.markdown(f"❌ **Missing** ({doc.filename})")
                    if missing:
                        st.write(", ".join(missing))
                    else:
                        st.write("All keywords present!")
    
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")


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
    st.title("AI-Powered Resume Screening Assistant")
    st.write(
        "This bachelor-thesis prototype ranks uploaded resumes against a job description using baseline IR models, "
        "semantic embedding models, and an interpretable hybrid score."
    )
    st.warning(
        "The system supports HR screening. It does not make final hiring decisions and should be reviewed by a human recruiter."
    )

    st.subheader("System Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            A["Resume/JD Upload"] --> B["Text Extraction"]
            B --> C["Cleaning and Skill Extraction"]
            C --> D["Model Scoring"]
            D --> E["Hybrid Score"]
            E --> F["Candidate Ranking"]
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
    st.title("Resume Matching")
    parser = DocumentParser()

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste job description", value=st.session_state.get("jd_text", ""), height=220)
        jd_file = st.file_uploader("Upload JD file", type=["pdf", "docx", "txt"], key="jd_file")
        if jd_file:
            try:
                jd_text = parse_uploaded_text(jd_file, parser)
                st.success(f"Loaded {jd_file.name}")
            except Exception as exc:
                st.error(str(exc))
        if st.button("Load sample data"):
            jd_text, sample_resumes = load_sample_documents()
            st.session_state["resume_docs"] = sample_resumes
            st.session_state["jd_text"] = jd_text
            st.rerun()
        st.session_state["jd_text"] = jd_text

    with right:
        st.subheader("Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload resume files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_files",
        )
        if uploaded_resumes:
            docs: list[ResumeDocument] = []
            for idx, uploaded in enumerate(uploaded_resumes, start=1):
                try:
                    text = parse_uploaded_text(uploaded, parser)
                    docs.append(ResumeDocument(candidate_id=f"candidate_{idx}", filename=uploaded.name, text=text))
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")
            if docs:
                st.session_state["resume_docs"] = docs

        resume_docs = st.session_state.get("resume_docs", [])
        st.write(f"Loaded resumes: {len(resume_docs)}")

    with st.expander("Text extraction preview", expanded=False):
        st.markdown("**Job description preview**")
        st.text(shorten_text(st.session_state.get("jd_text", ""), 1500))
        for doc in st.session_state.get("resume_docs", []):
            st.markdown(f"**{doc.filename}**")
            st.text(shorten_text(doc.text, 1000))

    # Display extracted keywords section
    display_extracted_keywords()

    st.subheader("Model and Scoring")
    model_label = st.selectbox(
        "Matching model",
        options=list(MODEL_NAMES.keys()),
        format_func=lambda key: MODEL_NAMES[key],
        index=0,
    )
    use_hybrid = st.toggle("Use hybrid score", value=True)
    weights = make_weight_controls()

    if st.button("Run matching", type="primary"):
        try:
            with st.spinner("Scoring resumes..."):
                st.session_state["results"] = match_resumes(
                    jd_text=st.session_state.get("jd_text", ""),
                    resumes=st.session_state.get("resume_docs", []),
                    model_key=model_label,
                    weights=weights,
                    use_hybrid_score=use_hybrid,
                )
                st.session_state["explanations"] = {}
                st.session_state["chat_history"] = []
        except Exception as exc:
            st.error(str(exc))

    results = st.session_state.get("results", [])
    if not results:
        return

    st.subheader("Ranked Candidates")
    table = pd.DataFrame([result.to_export_dict() for result in results])
    visible_columns = [
        "rank",
        "filename",
        "final_score",
        "semantic_score",
        "recommendation",
        "matched_skills",
        "missing_skills",
        "runtime_seconds",
    ]
    st.dataframe(table[visible_columns], use_container_width=True, hide_index=True)
    st.plotly_chart(candidate_score_bar(results), use_container_width=True)

    csv_bytes = results_to_csv_bytes(results)
    st.download_button("Export ranking CSV", data=csv_bytes, file_name="resume_screening_results.csv", mime="text/csv")
    report = screening_report_markdown(st.session_state["jd_text"], results)
    st.download_button("Export screening report", data=report, file_name="screening_report.md", mime="text/markdown")

    st.subheader("Candidate Detail Analysis")
    selected = st.selectbox(
        "Candidate",
        options=[result.filename for result in results],
        index=0,
    )
    result = next(item for item in results if item.filename == selected)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{result.final_score:.2f}%")
    c2.metric("Semantic score", f"{result.semantic_score:.2f}%")
    c3.metric("Resume years", f"{result.detected_resume_years:g}")
    c4.metric("Required years", f"{result.required_years:g}")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**Matched skills**")
        st.write(", ".join(result.matched_skills) or "None detected")
        st.markdown("**Education/certification evidence**")
        evidence = result.education_signals + result.certification_signals
        st.write(", ".join(evidence) or "No dictionary evidence detected")
    with detail_right:
        st.markdown("**Missing required skills**")
        st.write(", ".join(result.missing_skills) or "None detected")
        st.markdown("**Recommendation**")
        st.write(result.recommendation)

    if st.button("Generate explanation"):
        explanation = generate_candidate_explanation(result, st.session_state["jd_text"], result.cleaned_resume_text, use_llm=True)
        st.session_state["explanations"][result.filename] = explanation
    if result.filename in st.session_state["explanations"]:
        st.markdown(st.session_state["explanations"][result.filename])

    st.subheader("Chatbot")
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ask about candidate ranking, missing skills, or top candidates")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        answer = answer_hr_question(question, results, st.session_state["jd_text"], use_llm=True)
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
    st.title("Qdrant RAG Vector Store")
    st.write(
        "This page stores cleaned JD and resume chunks as embedding vectors in Qdrant. "
        "The retrieved chunks can be used as grounded context for a future RAG explanation or HR question-answering module."
    )

    jd_text = st.session_state.get("jd_text", "")
    resume_docs: list[ResumeDocument] = st.session_state.get("resume_docs", [])
    if not jd_text or not resume_docs:
        st.info("Load or upload documents on the Screening page first, or use the sample data button below.")
        if st.button("Load sample documents for RAG"):
            jd_text, resume_docs = load_sample_documents()
            st.session_state["jd_text"] = jd_text
            st.session_state["resume_docs"] = resume_docs
            st.rerun()
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(resume_docs) + 1)
    c2.metric("Collection", QDRANT_COLLECTION_NAME)
    c3.metric("Embedding model", DEFAULT_RAG_EMBEDDING_MODEL.split("/")[-1])

    if QDRANT_URL:
        st.caption(f"Qdrant server: {QDRANT_URL}")
    else:
        st.caption(f"Qdrant local path: {QDRANT_LOCAL_PATH}")

    if st.button("Index current JD and resumes", type="primary"):
        try:
            with st.spinner("Embedding chunks and writing vectors to Qdrant..."):
                store = QdrantRAGStore()
                count = store.index_documents(build_rag_documents(jd_text, resume_docs))
            st.success(f"Indexed {count} chunks into Qdrant collection '{QDRANT_COLLECTION_NAME}'.")
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Retrieve RAG Context")
    query = st.text_input("Search query", value="Which candidates show Python, NLP, and cloud deployment experience?")
    limit = st.slider("Top chunks", min_value=1, max_value=10, value=5)
    document_type = st.selectbox("Document type filter", ["All", "resume", "job_description"])
    if st.button("Search Qdrant"):
        try:
            with st.spinner("Retrieving nearest chunks..."):
                store = QdrantRAGStore()
                st.session_state["rag_results"] = store.search(
                    query,
                    limit=limit,
                    document_type=None if document_type == "All" else document_type,
                )
        except Exception as exc:
            st.error(str(exc))

    results = st.session_state.get("rag_results", [])
    if not results:
        return

    rows = [
        {
            "score": round(result.score, 4),
            "type": result.document_type,
            "filename": result.filename,
            "chunk_id": result.chunk_id,
            "preview": shorten_text(result.text, 220),
        }
        for result in results
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**RAG context block**")
    st.code(build_rag_context(results), language="text")


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
    page = st.sidebar.radio("Navigation", ["Home", "Screening", "Evaluation", "RAG Vector Store", "Dataset & Thesis"])
    if page == "Home":
        home_page()
    elif page == "Screening":
        screening_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "RAG Vector Store":
        rag_page()
    else:
        dataset_page()


if __name__ == "__main__":
    main()
