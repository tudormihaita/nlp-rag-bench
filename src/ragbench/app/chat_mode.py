# Author: Tudor Mihaita
from collections.abc import Mapping

import streamlit as st

from ragbench.app.components import render_trace
from ragbench.pipeline import RAGPipeline


def render_chat_mode(
    pipelines: Mapping[str, RAGPipeline | None],
    method: str,
    top_k: int,
) -> None:
    st.title("RAGBench — Multi-Hop QA")

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

            with st.spinner("Thinking..."):
                answer_stream, passages, trace = pipeline.stream(question, top_k=top_k)

            if passages:
                with st.expander(f"Retrieved context ({len(passages)} passages)"):
                    for p in passages:
                        st.markdown(f"**{p.title}** - score: `{p.score:.3f}`")
                        st.caption(p.text)

            if trace.sub_queries or trace.rerank_scores or trace.notes:
                with st.expander("Reasoning trace"):
                    render_trace(trace, passages)

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
