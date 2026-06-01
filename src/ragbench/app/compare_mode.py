# Author: Tudor Mihaita
from collections.abc import Mapping

import streamlit as st

from ragbench.app.components import render_trace
from ragbench.pipeline import PipelineResult, RAGPipeline


def _render_results(
    results: dict[str, PipelineResult], pipelines: Mapping[str, RAGPipeline | None]
) -> None:
    active = [m for m in pipelines if pipelines[m] is not None]
    cols = st.columns(len(active))
    for col, method in zip(cols, active):
        result = results[method]
        with col:
            st.subheader(method)
            st.markdown(result.answer)
            if result.passages:
                with st.expander(f"Context ({len(result.passages)} passages)"):
                    for p in result.passages:
                        st.markdown(f"**{p.title}** — `{p.score:.3f}`")
                        st.caption(p.text)
            if result.trace.sub_queries or result.trace.rerank_scores or result.trace.notes:
                with st.expander("Reasoning trace"):
                    render_trace(result.trace, result.passages)


def render_compare_mode(pipelines: Mapping[str, RAGPipeline | None], top_k: int) -> None:
    st.header("Compare methods")
    st.caption("Run the same question through all methods side-by-side.")

    if "compare" in st.session_state:
        saved = st.session_state["compare"]
        st.caption(f"Results for: *{saved['question']}*")
        _render_results(saved["results"], pipelines)

    question = st.chat_input("Ask a multi-hop question to compare all methods…")

    if question:
        active = [(m, p) for m, p in pipelines.items() if p is not None]
        results: dict[str, PipelineResult] = {}
        cols = st.columns(len(active))

        for col, (method, pipeline) in zip(cols, active):
            with col:
                with st.status(f"Running {method}…", expanded=False) as status:
                    result = pipeline.run(question, top_k=top_k)
                    status.update(label=method, state="complete")
                results[method] = result
                st.markdown(result.answer)
                if result.passages:
                    with st.expander(f"Context ({len(result.passages)} passages)"):
                        for p in result.passages:
                            st.markdown(f"**{p.title}** — `{p.score:.3f}`")
                            st.caption(p.text)
                if result.trace.sub_queries or result.trace.rerank_scores or result.trace.notes:
                    with st.expander("Reasoning trace"):
                        render_trace(result.trace, result.passages)

        st.session_state["compare"] = {"question": question, "results": results}
