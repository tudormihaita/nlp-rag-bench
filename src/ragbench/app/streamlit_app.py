import streamlit as st

st.set_page_config(page_title="RAGBench", page_icon="🔍", layout="wide")


@st.cache_resource
def load_pipelines():
    from ragbench.config import settings
    from ragbench.generation.llm import Generator
    from ragbench.indexing.builder import index_exists, load_collection
    from ragbench.indexing.embedder import Embedder
    from ragbench.pipeline import RAGPipeline
    from ragbench.retrievers.decomposition import DecompositionRetriever
    from ragbench.retrievers.naive import NaiveRetriever
    from ragbench.retrievers.reranking import ReRankRetriever

    llm = Generator(settings.generator_model)
    pipelines: dict[str, RAGPipeline | None] = {
        "No-RAG": RAGPipeline(generator=llm, retriever=None, top_k=settings.top_k),
    }

    if not index_exists(settings.embedding_model, str(settings.chroma_db_dir)):
        pipelines |= {"Classic RAG": None, "Re-ranking": None, "Decomposition": None}
        return pipelines

    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    collection = load_collection(str(settings.embedding_model), str(settings.chroma_db_dir))
    pipelines |= {
        "Classic RAG": RAGPipeline(
            generator=llm,
            retriever=NaiveRetriever(collection, embedder),
            top_k=settings.top_k,
        ),
        "Re-ranking": RAGPipeline(
            generator=llm,
            retriever=ReRankRetriever(
                collection, embedder,
                reranker_model=settings.reranker_model,
                device=settings.reranker_device,
                candidate_k=settings.rerank_candidate_k,
            ),
            top_k=settings.top_k,
        ),
        "Decomposition": RAGPipeline(
            generator=llm,
            retriever=DecompositionRetriever(collection, embedder, llm),
            top_k=settings.top_k,
        ),
    }
    return pipelines


@st.cache_data
def load_benchmark_data():
    import json

    from ragbench.config import settings

    paths = [
        settings.processed_dir / "sampled_questions.json",
        settings.processed_dir / "gold_index.json",
        settings.processed_dir / "pooled_corpus.json",
    ]
    if not all(p.exists() for p in paths):
        return None, None, None

    questions = json.loads(paths[0].read_text())
    gold_index = json.loads(paths[1].read_text())
    corpus = json.loads(paths[2].read_text())
    doc_lookup = {p["doc_id"]: p for p in corpus}
    return questions, gold_index, doc_lookup


from ragbench.app.benchmark_mode import render_benchmark_mode
from ragbench.app.chat_mode import render_chat_mode
from ragbench.app.compare_mode import render_compare_mode

# Fail fast with a readable message instead of a raw connection traceback
try:
    import ollama as _ollama
    _ollama.list()
except Exception:
    st.error(
        "**Ollama is not running.** Start it with `ollama serve` then refresh this page.",
        icon="🔴",
    )
    st.stop()

pipelines = load_pipelines()

if pipelines.get("Classic RAG") is None:
    st.info(
        "Vector index not found. Only **No-RAG** is available. "
        "Build document corpus first to enable the retrieval methods.",
        icon="ℹ️",
    )

questions, gold_index, doc_lookup = load_benchmark_data()

chat_tab, compare_tab, benchmark_tab = st.tabs(["Chat", "Compare", "Benchmark"])

with chat_tab:
    render_chat_mode(pipelines)

with compare_tab:
    render_compare_mode(pipelines)

with benchmark_tab:
    if questions is None:
        st.warning("Preprocess and sample data first to enable Benchmark mode.")
    else:
        render_benchmark_mode(pipelines, questions, gold_index, doc_lookup)
