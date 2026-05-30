import chromadb

from ragbench.indexing.embedder import Embedder

from .base import RetrievedPassage, RetrievalTrace


class NaiveRetriever:
    """Classic dense retrieval: embed query, return top-k by cosine similarity"""

    def __init__(self, collection: chromadb.Collection, embedder: Embedder) -> None:
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        q_emb = self.embedder.encode_query(query)
        res = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=k)

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
        return passages, RetrievalTrace()
