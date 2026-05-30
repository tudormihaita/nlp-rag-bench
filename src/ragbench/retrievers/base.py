from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    title: str
    score: float  # cosine similarity [-1, 1]; higher is better


@dataclass
class RetrievalTrace:
    """Optional metadata for the UI to render (sub-questions, rerank scores, etc.)"""
    sub_queries: list[str] = field(default_factory=list)
    intermediate_answers: list[str] = field(default_factory=list)
    enriched_queries: list[str] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    notes: str = ""


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        ...