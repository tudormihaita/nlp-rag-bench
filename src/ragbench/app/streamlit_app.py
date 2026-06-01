# Author: Tudor Mihaita
from collections.abc import Mapping

import streamlit as st

from ragbench.factory import PipelineName
from ragbench.pipeline import RAGPipeline

st.set_page_config(page_title="RAGBench", page_icon="🔍", layout="wide")


@st.cache_resource
def load_pipelines() -> Mapping[str, RAGPipeline | None]:
    from ragbench.config import settings
    from ragbench.factory import build_pipelines
    from ragbench.generation.llm import Generator
    from ragbench.indexing.builder import index_exists
    from ragbench.indexing.embedder import Embedder

    gen = Generator(
        settings.generator_model,
        host=settings.api_url,
        auth_bearer=settings.api_auth_bearer,
        api_src=settings.api_src,
    )

    if not index_exists(settings.embedding_model, str(settings.chroma_db_dir)):
        return {
            PipelineName.NO_RAG: RAGPipeline(generator=gen, retriever=None, top_k=settings.top_k),
            PipelineName.CLASSIC_RAG: None,
            PipelineName.RERANKING: None,
            PipelineName.DECOMPOSITION: None,
            PipelineName.ITERATIVE_DECOMPOSITION: None,
        }

    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    return build_pipelines(embedder, gen, settings, include_iterative=True)


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
from ragbench.config import settings
from ragbench.generation.llm import Generator

# Fail fast with a readable message instead of a raw connection traceback
_llm = Generator(
    settings.generator_model,
    host=settings.api_url,
    auth_bearer=settings.api_auth_bearer,
    api_src=settings.api_src,
)
if not _llm.health_check():
    if settings.api_src == "ollama":
        msg = "**Ollama is not running.** Start it with `ollama serve` then refresh this page."
    else:
        msg = f"**LLM backend is not reachable** (`API {settings.api_url}`). Check that the server is running and authentication is set correctly if required."
    st.error(msg, icon="🔴")
    st.stop()

del _llm  # don't keep the temporary instance around

pipelines = load_pipelines()

if pipelines.get(PipelineName.CLASSIC_RAG) is None:
    st.info(
        "Vector index not found. Only **No-RAG** is available. "
        "Build document corpus first to enable the retrieval methods.",
        icon="ℹ️",
    )

questions, gold_index, doc_lookup = load_benchmark_data()

# Global sidebar — visible on all tabs
with st.sidebar:
    st.header("Settings")
    active_methods = [m for m, p in pipelines.items() if p is not None]
    chat_method = st.selectbox("Retrieval method", active_methods, help="Used in the Chat tab")
    top_k = st.slider("Top-k passages", min_value=1, max_value=20, value=settings.top_k)
    st.divider()
    _gen_pipeline = pipelines.get(PipelineName.CLASSIC_RAG) or next(
        (p for p in pipelines.values() if p is not None), None
    )
    if _gen_pipeline:
        st.caption(f"Generator: {_gen_pipeline.generator.model}")

chat_tab, compare_tab, benchmark_tab = st.tabs(["Chat", "Compare", "Benchmark"])

with chat_tab:
    render_chat_mode(pipelines, chat_method, top_k)

with compare_tab:
    render_compare_mode(pipelines, top_k)

with benchmark_tab:
    if questions is None:
        st.warning("Preprocess and sample data first to enable Benchmark mode.")
    else:
        render_benchmark_mode(pipelines, questions, gold_index, doc_lookup, top_k)
