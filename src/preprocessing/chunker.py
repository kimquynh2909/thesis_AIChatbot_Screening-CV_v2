from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[TextChunk]:
    """Split long documents into overlapping word chunks for embedding models."""
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= max_words:
        return [TextChunk(chunk_id=0, text=" ".join(words))]

    chunks: list[TextChunk] = []
    start = 0
    chunk_id = 0
    step = max(1, max_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(TextChunk(chunk_id=chunk_id, text=" ".join(words[start:end])))
        if end == len(words):
            break
        start += step
        chunk_id += 1
    return chunks


def aggregate_chunk_scores(scores: list[float], strategy: str = "mean_top3") -> float:
    if not scores:
        return 0.0
    ordered = sorted(scores, reverse=True)
    if strategy == "max":
        return float(ordered[0])
    if strategy == "mean_top3":
        top = ordered[:3]
        return float(sum(top) / len(top))
    return float(sum(scores) / len(scores))
