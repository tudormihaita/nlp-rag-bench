# Author: Alexandru Profir
import math
from dataclasses import replace

import chromadb
from sentence_transformers import CrossEncoder

from ragbench.indexing.embedder import Embedder

from .base import RetrievalTrace, RetrievedPassage
from .naive import NaiveRetriever


def _safe_device(requested: str) -> str:
    if requested == "mps":
        try:
            import torch

            if not torch.backends.mps.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return requested


class ReRankRetriever:
    """
    Dense retrieval over a large candidate pool, then cross-encoder re-ranking.
    Fetches `candidate_k` passages with the bi-encoder, scores every
    (query, passage) pair with a cross-encoder, returns the top-k by rerank score.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        reranker_model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        candidate_k: int = 40,
    ) -> None:
        self.naive = NaiveRetriever(collection, embedder)
        self.reranker = CrossEncoder(reranker_model, device=_safe_device(device))
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        # diversify=False: let the CrossEncoder decide relevance over the full candidate pool
        candidates, _ = self.naive.retrieve(query, k=self.candidate_k, diversify=False)

        pairs = [[query, p.text] for p in candidates]
        scores: list[float] = self.reranker.predict(pairs).tolist()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]

        # Sigmoid maps unbounded CrossEncoder logits to [0, 1] for interpretable scores
        passages = [replace(p, score=1.0 / (1.0 + math.exp(-float(s)))) for s, p in ranked]
        rerank_scores = [1.0 / (1.0 + math.exp(-float(s))) for s, _ in ranked]

        return passages, RetrievalTrace(
            rerank_scores=rerank_scores,
            notes=f"Scored {len(candidates)} candidates with cross-encoder; retained top {k}.",
        )
