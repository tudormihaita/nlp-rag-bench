import streamlit as st

from ragbench.pipeline import PipelineResult, RAGPipeline


def _render_results(results: dict[str, PipelineResult], pipelines: dict[str, RAGPipeline | None]) -> None:
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
            if result.trace.sub_queries:
                with st.expander("Reasoning trace"):
                    for sq in result.trace.sub_queries:
                        st.write(f"• {sq}")
            if result.trace.notes:
                st.caption(f"Note: {result.trace.notes}")


def render_compare_mode(pipelines: dict[str, RAGPipeline | None]) -> None:
    st.header("Compare methods")
    st.caption("Run the same question through all methods side-by-side.")

    question = st.text_input(
        "Question",
        placeholder="Who is the spouse of the performer of Imagine?",
        key="compare_input",
    )
    run = st.button("Run comparison", type="primary", disabled=not question)

    if run and question:
        active = [(m, p) for m, p in pipelines.items() if p is not None]
        results: dict[str, PipelineResult] = {}
        cols = st.columns(len(active))

        # Run methods sequentially; spinner shows in each column while it's running
        for col, (method, pipeline) in zip(cols, active):
            with col:
                st.subheader(method)
                with st.spinner("Running…"):
                    results[method] = pipeline.run(question)

        st.session_state["compare"] = {"question": question, "results": results}
        st.rerun()

    if "compare" in st.session_state:
        saved = st.session_state["compare"]
        if saved["question"] == question:
            st.caption(f"Results for: *{saved['question']}*")
            _render_results(saved["results"], pipelines)
