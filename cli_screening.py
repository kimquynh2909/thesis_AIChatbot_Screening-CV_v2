#!/usr/bin/env python
"""CLI-based resume screening tool without Streamlit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DEFAULT_HYBRID_WEIGHTS,
    MODEL_NAMES,
    SAMPLE_JD_DIR,
    SAMPLE_RESUME_DIR,
)
from src.preprocessing.document_parser import DocumentParser
from src.preprocessing.skill_extractor import SkillExtractor
from src.services.matching_service import (
    MatchResult,
    ResumeDocument,
    match_resumes,
)
from src.utils.file_utils import read_file


def load_documents(jd_path: str, resume_paths: list[str]) -> tuple[str, list[ResumeDocument]]:
    """Load JD and resume documents."""
    parser = DocumentParser()
    
    # Load JD
    jd_text = read_file(jd_path)
    
    # Load resumes
    resumes = []
    for resume_path in resume_paths:
        path = Path(resume_path)
        text = read_file(resume_path)
        resumes.append(
            ResumeDocument(
                candidate_id=path.stem,
                filename=path.name,
                text=text,
            )
        )
    
    return jd_text, resumes


def print_results(results: list[MatchResult], verbose: bool = False) -> None:
    """Print ranking results in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Rank':<6} {'Candidate':<30} {'Model':<20} {'Semantic':<12} {'Final':<12} {'Recommendation':<15}")
    print("=" * 100)
    
    for result in results:
        print(
            f"{result.rank:<6} {result.candidate_id:<30} {result.model_name:<20} "
            f"{result.semantic_score:<12.2f} {result.final_score:<12.2f} {result.recommendation:<15}"
        )
    
    print("=" * 100)
    
    if verbose:
        print("\n" + "=" * 100)
        print("DETAILED BREAKDOWN")
        print("=" * 100)
        for result in results:
            print(f"\n📋 {result.candidate_id.upper()}")
            print(f"   Rank: {result.rank}")
            print(f"   Final Score: {result.final_score:.2f}/100")
            print(f"   Recommendation: {result.recommendation}")
            print(f"   Runtime: {result.runtime_seconds:.4f}s")
            print(f"\n   Skill Analysis:")
            print(f"      Matched Skills: {', '.join(result.matched_skills) if result.matched_skills else 'None'}")
            print(f"      Missing Skills: {', '.join(result.missing_skills) if result.missing_skills else 'None'}")
            print(f"      Resume has {len(result.resume_skills)} skills")
            print(f"      JD requires {len(result.jd_skills)} skills")
            print(f"\n   Experience:")
            print(f"      Detected: {result.detected_resume_years:.1f} years")
            print(f"      Required: {result.required_years:.1f} years")
            print(f"\n   Education/Certification:")
            print(f"      Signals: {', '.join(result.education_signals) if result.education_signals else 'None'}")
            if result.certification_signals:
                print(f"      Certifications: {', '.join(result.certification_signals)}")
            print(f"\n   Score Breakdown:")
            print(f"      Semantic Similarity: {result.breakdown.semantic_score:.2f}%")
            print(f"      Skill Match: {result.breakdown.skill_score:.2f}%")
            print(f"      Experience Match: {result.breakdown.experience_score:.2f}%")
            print(f"      Education Match: {result.breakdown.education_score:.2f}%")


def export_results(results: list[MatchResult], output_path: str) -> None:
    """Export results to CSV."""
    data = [result.to_export_dict() for result in results]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Resume Screening Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use sample documents with SBERT
  python cli_screening.py --model sbert

  # Screen specific resumes against a JD
  python cli_screening.py --jd jobs/job.txt --resumes resumes/resume1.txt resumes/resume2.txt --model e5

  # Use custom weights for hybrid scoring
  python cli_screening.py --model hybrid --semantic 0.6 --skills 0.2 --experience 0.1 --education 0.1

  # Verbose output with detailed breakdowns
  python cli_screening.py --model bm25 --verbose

  # Export results to CSV
  python cli_screening.py --model tfidf --output results.csv
        """,
    )
    
    # Document inputs
    parser.add_argument(
        "--jd",
        type=str,
        default=None,
        help="Path to job description file (default: sample JD)",
    )
    parser.add_argument(
        "--resumes",
        type=str,
        nargs="+",
        default=None,
        help="Paths to resume files (default: sample resumes)",
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_NAMES.keys()),
        default="sbert",
        help=f"Matching model to use (default: sbert). Options: {', '.join(MODEL_NAMES.keys())}",
    )
    
    # Hybrid scoring weights
    parser.add_argument(
        "--semantic",
        type=float,
        default=DEFAULT_HYBRID_WEIGHTS["semantic"],
        help=f"Weight for semantic similarity (default: {DEFAULT_HYBRID_WEIGHTS['semantic']})",
    )
    parser.add_argument(
        "--skills",
        type=float,
        default=DEFAULT_HYBRID_WEIGHTS["skills"],
        help=f"Weight for skill matching (default: {DEFAULT_HYBRID_WEIGHTS['skills']})",
    )
    parser.add_argument(
        "--experience",
        type=float,
        default=DEFAULT_HYBRID_WEIGHTS["experience"],
        help=f"Weight for experience matching (default: {DEFAULT_HYBRID_WEIGHTS['experience']})",
    )
    parser.add_argument(
        "--education",
        type=float,
        default=DEFAULT_HYBRID_WEIGHTS["education"],
        help=f"Weight for education matching (default: {DEFAULT_HYBRID_WEIGHTS['education']})",
    )
    
    # Output options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed score breakdowns for each candidate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    
    args = parser.parse_args()
    
    # Load documents
    print("📄 Loading documents...")
    
    if args.jd:
        jd_path = args.jd
    else:
        jd_files = sorted(SAMPLE_JD_DIR.glob("*.txt"))
        if not jd_files:
            print("❌ Error: No sample JD found. Provide --jd argument.")
            sys.exit(1)
        jd_path = str(jd_files[0])
        print(f"   Using sample JD: {jd_files[0].name}")
    
    if args.resumes:
        resume_paths = args.resumes
    else:
        resume_files = sorted(SAMPLE_RESUME_DIR.glob("*.txt"))
        if not resume_files:
            print("❌ Error: No sample resumes found. Provide --resumes argument.")
            sys.exit(1)
        resume_paths = [str(f) for f in resume_files]
        print(f"   Using {len(resume_files)} sample resumes")
    
    jd_text, resumes = load_documents(jd_path, resume_paths)
    print(f"✅ Loaded 1 JD and {len(resumes)} resume(s)")
    
    # Run matching
    print(f"\n🔍 Screening with {MODEL_NAMES[args.model]}...")
    
    weights = {
        "semantic": args.semantic,
        "skills": args.skills,
        "experience": args.experience,
        "education": args.education,
    }
    
    skill_extractor = SkillExtractor()
    results = match_resumes(
        jd_text=jd_text,
        resumes=resumes,
        model_key=args.model,
        weights=weights,
        use_hybrid_score=True,
        skill_extractor=skill_extractor,
    )
    
    # Display results
    if args.json:
        output_data = [
            {
                "rank": r.rank,
                "candidate_id": r.candidate_id,
                "filename": r.filename,
                "model_name": r.model_name,
                "semantic_score": r.semantic_score,
                "final_score": r.final_score,
                "recommendation": r.recommendation,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "detected_years": r.detected_resume_years,
                "required_years": r.required_years,
                "runtime_seconds": r.runtime_seconds,
            }
            for r in results
        ]
        print(json.dumps(output_data, indent=2))
    else:
        print_results(results, verbose=args.verbose)
    
    # Export if requested
    if args.output:
        export_results(results, args.output)
    
    print(f"\n✨ Done! Screening completed in {sum(r.runtime_seconds for r in results):.4f}s total")


if __name__ == "__main__":
    main()
