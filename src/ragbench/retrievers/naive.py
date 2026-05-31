import chromadb

from ragbench.indexing.embedder import Embedder

from .base import RetrievalTrace, RetrievedPassage

_MAX_PER_TITLE = 2  # max passages from the same article title in final top-k
_FETCH_MULT = 3  # overfetch factor to ensure k diverse results after filtering


def _diversify(
    passages: list[RetrievedPassage], max_per_title: int = _MAX_PER_TITLE
) -> list[RetrievedPassage]:
    """Keep at most max_per_title passages per article title, preserving score order."""
    seen: dict[str, int] = {}
    out = []
    for p in passages:
        count = seen.get(p.title, 0)
        if count < max_per_title:
            out.append(p)
            seen[p.title] = count + 1
    return out


class NaiveRetriever:
    """Classic dense retrieval: embed query, return top-k by cosine similarity.

    Internally fetches _FETCH_MULT * k candidates and applies title diversification
    so that passages from the same article don't crowd out topically distinct results.
    """

    def __init__(self, collection: chromadb.Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(
        self, query: str, k: int = 5, diversify: bool = True
    ) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        q_emb = self.embedder.encode_query(query)
        fetch_k = k * _FETCH_MULT if diversify else k
        res = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=fetch_k)

        passages = [
            RetrievedPassage(
                doc_id=doc_id,
                text=text,
                title=meta["title"],
                score=1.0 - distance,  # cosine distance -> cosine similarity
            )
            for doc_id, text, meta, distance in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            )
        ]
        if diversify:
            passages = _diversify(passages)
        return passages[:k], RetrievalTrace()
