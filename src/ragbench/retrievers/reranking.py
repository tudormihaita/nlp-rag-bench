import chromadb
from sentence_transformers import CrossEncoder

from ragbench.indexing.embedder import Embedder

from .base import RetrievedPassage, RetrievalTrace
from .naive import NaiveRetriever


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
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        candidate_k: int = 30,
    ) -> None:
        self.naive = NaiveRetriever(collection, embedder)
        self.reranker = CrossEncoder(reranker_model, device=device)
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        candidates, _ = self.naive.retrieve(query, k=self.candidate_k)

        pairs = [[query, p.text] for p in candidates]
        scores: list[float] = self.reranker.predict(pairs).tolist()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]

        passages = []
        rerank_scores = []
        for score, passage in ranked:
            passage.score = score  # replace cosine similarity with rerank logit
            passages.append(passage)
            rerank_scores.append(score)

        return passages, RetrievalTrace(rerank_scores=rerank_scores)
