from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.matching_service import MatchResult


def candidate_score_bar(results: list[MatchResult]) -> go.Figure:
    frame = pd.DataFrame(
        {
            "Candidate": [result.filename for result in results],
            "Final Score": [result.final_score for result in results],
            "Semantic Score": [result.semantic_score for result in results],
        }
    )
    fig = px.bar(
        frame,
        x="Candidate",
        y=["Final Score", "Semantic Score"],
        barmode="group",
        title="Candidate Matching Scores",
    )
    fig.update_layout(xaxis_tickangle=-30, yaxis_range=[0, 100], legend_title_text="Score Type")
    return fig


def model_comparison_chart(comparison: pd.DataFrame, metric: str = "ndcg@5") -> go.Figure:
    if comparison.empty or metric not in comparison.columns:
        return go.Figure()
    fig = px.bar(comparison, x="model", y=metric, title=f"Model Comparison by {metric.upper()}")
    fig.update_layout(yaxis_range=[0, 1])
    return fig


def similarity_distribution(scored_pairs: pd.DataFrame) -> go.Figure:
    if scored_pairs.empty or "score" not in scored_pairs.columns:
        return go.Figure()
    fig = px.histogram(scored_pairs, x="score", color="model" if "model" in scored_pairs.columns else None, nbins=30)
    fig.update_layout(title="Similarity Score Distribution")
    return fig
