from ragbench.config import Settings
from ragbench.generation.llm import Generator
from ragbench.indexing.embedder import Embedder
from ragbench.indexing.builder import load_collection
from ragbench.pipeline import RAGPipeline
from ragbench.retrievers.naive import NaiveRetriever
from ragbench.retrievers.reranking import ReRankRetriever
from ragbench.retrievers.decomposition import DecompositionRetriever


def build_pipelines(
    embedder: Embedder,
    gen: Generator,
    settings: Settings,
    include_iterative: bool = True,
) -> dict[str, RAGPipeline]:
    """Construct all RAG pipelines from a shared embedder and generator."""
    collection = load_collection(settings.embedding_model, str(settings.chroma_db_dir))

    pipelines: dict[str, RAGPipeline] = {
        "No-RAG": RAGPipeline(generator=gen, retriever=None, top_k=settings.top_k),
        "Classic RAG": RAGPipeline(
            generator=gen,
            retriever=NaiveRetriever(collection, embedder),
            top_k=settings.top_k,
        ),
        "Re-ranking": RAGPipeline(
            generator=gen,
            retriever=ReRankRetriever(
                collection,
                embedder,
                reranker_model=settings.reranker_model,
                device=settings.reranker_device,
                candidate_k=settings.rerank_candidate_k,
            ),
            top_k=settings.top_k,
        ),
        "Decomposition": RAGPipeline(
            generator=gen,
            retriever=DecompositionRetriever(collection, embedder, gen),
            top_k=settings.top_k,
        ),
    }

    if include_iterative:
        pipelines["Iterative Decomposition"] = RAGPipeline(
            generator=gen,
            retriever=DecompositionRetriever(collection, embedder, gen, iterative=True),
            top_k=settings.top_k,
        )

    return pipelines
