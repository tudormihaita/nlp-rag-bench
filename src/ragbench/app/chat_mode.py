import streamlit as st

from ragbench.pipeline import RAGPipeline


def render_chat_mode(pipelines: dict[str, RAGPipeline | None]) -> None:
    st.title("RAGBench — Multi-Hop QA")

    with st.sidebar:
        st.header("Settings")
        method = st.selectbox(
            "Retrieval method",
            list(pipelines.keys()),
            help="Re-ranking and Decomposition RAG methods are available.",
        )
        top_k = st.slider("Top-k passages", min_value=1, max_value=10, value=5)
        st.divider()
        st.caption(
            "Generator: "
            + (pipelines.get("Classic RAG") or next(iter(pipelines.values()))).generator.model
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("passages"):
                with st.expander(f"Retrieved context ({len(msg['passages'])} passages)"):
                    for p in msg["passages"]:
                        st.markdown(f"**{p['title']}** - score: `{p['score']:.3f}`")
                        st.caption(p["text"])

    if question := st.chat_input("Ask a multi-hop question…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            pipeline = pipelines[method]
            if pipeline is None:
                st.warning(f"**{method}** is not yet implemented.")
                return

            pipeline.top_k = top_k
            answer_stream, passages, trace = pipeline.stream(question)

            if passages:
                with st.expander(f"Retrieved context ({len(passages)} passages)"):
                    for p in passages:
                        st.markdown(f"**{p.title}** - score: `{p.score:.3f}`")
                        st.caption(p.text)

            if trace.sub_queries:
                with st.expander("Reasoning trace"):
                    for sq in trace.sub_queries:
                        st.write(f"• {sq}")

            answer = st.write_stream(answer_stream)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "passages": [
                    {"title": p.title, "text": p.text, "score": p.score} for p in passages
                ],
            }
        )
