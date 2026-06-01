# Author: Alexandru Profir
import random
from collections.abc import Mapping
from typing import Any

import streamlit as st

from ragbench.app.components import render_trace

from ragbench.evaluation.metrics import (
    all_recall_at_k,
    exact_match,
    hit_at_k,
    mrr,
    recall_at_k,
    token_f1,
)
from ragbench.pipeline import PipelineResult, RAGPipeline


def _compute_metrics(result: PipelineResult, references: list[str], gold_ids: list[str]) -> dict:
    retrieved_ids = [p.doc_id for p in result.passages]
    return {
        "EM": exact_match(result.answer, references),
        "F1": token_f1(result.answer, references),
        "Hit@k": hit_at_k(retrieved_ids, gold_ids) if retrieved_ids else None,
        "Recall@k": recall_at_k(retrieved_ids, gold_ids) if retrieved_ids else None,
        "All-Recall@k": all_recall_at_k(retrieved_ids, gold_ids) if retrieved_ids else None,
        "MRR": mrr(retrieved_ids, gold_ids) if retrieved_ids else None,
    }


def _pick_question(questions: list[dict], hop_filter: str) -> dict:
    if hop_filter == "All":
        pool = questions
    else:
        hop_n = int(hop_filter[0])
        pool = [q for q in questions if q["hop_count"] == hop_n]
    return random.choice(pool)


def render_benchmark_mode(
    pipelines: Mapping[str, RAGPipeline | None],
    questions: list[dict[str, Any]],
    gold_index: dict[str, list[str]],
    doc_lookup: dict[str, dict[str, Any]],
    top_k: int,
) -> None:
    st.header("Benchmark")
    st.caption("Step through individual MuSiQue questions and see how each method performs.")

    filter_col, btn_col = st.columns([3, 1])
    with filter_col:
        hop_filter = st.selectbox("Hop count", ["All", "2-hop", "3-hop", "4-hop"], key="bench_hop")
    with btn_col:
        run_btn = st.button("Run all methods", type="primary", use_container_width=True)
        new_q = st.button("New question", use_container_width=True)

    # Pick / refresh question
    hop_changed = hop_filter != st.session_state.get("bench_hop_prev")
    if new_q or hop_changed or "bench_q" not in st.session_state:
        st.session_state["bench_q"] = _pick_question(questions, hop_filter)
        st.session_state["bench_hop_prev"] = hop_filter
        st.session_state.pop("bench_results", None)

    q = st.session_state["bench_q"]
    gold_ids = gold_index.get(q["id"], [])
    references = [q["answer"]] + q.get("answer_aliases", [])

    # Question display
    st.divider()
    st.subheader(q["question"])
    st.caption(f"`{q['id']}` — {q['hop_count']}-hop ({q['hop_prefix']})")

    # Gold reveals
    rev_col1, rev_col2 = st.columns(2)
    with rev_col1:
        with st.expander("Reveal gold answer"):
            st.markdown(f"**{q['answer']}**")
            if q.get("answer_aliases"):
                st.caption("Also accepted: " + ", ".join(q["answer_aliases"]))
    with rev_col2:
        with st.expander(f"Reveal gold passages ({len(gold_ids)})"):
            for doc_id in gold_ids:
                passage = doc_lookup.get(doc_id)
                if passage:
                    st.markdown(f"**{passage['title']}**")
                    st.caption(passage["text"])
                else:
                    st.code(doc_id)

    # Run all methods
    if run_btn:
        active = [(m, p) for m, p in pipelines.items() if p is not None]
        results: dict[str, PipelineResult] = {}
        with st.spinner("Running all methods..."):
            for method, pipeline in active:
                results[method] = pipeline.run(q["question"], top_k=top_k)
        st.session_state["bench_results"] = results

    # Render results
    if "bench_results" in st.session_state:
        results = st.session_state["bench_results"]
        active_methods = [m for m in pipelines if m in results]

        # Metrics table
        st.divider()
        st.subheader("Metrics")
        table_rows = []
        for method in active_methods:
            result = results[method]
            m = _compute_metrics(result, references, gold_ids)
            table_rows.append(
                {
                    "Method": method,
                    "Answer": result.answer[:80] + ("…" if len(result.answer) > 80 else ""),
                    "EM": f"{m['EM']:.2f}",
                    "F1": f"{m['F1']:.2f}",
                    "Hit@k": f"{m['Hit@k']:.2f}" if m["Hit@k"] is not None else "—",
                    "Recall@k": f"{m['Recall@k']:.2f}" if m["Recall@k"] is not None else "—",
                    "All-Rec@k": f"{m['All-Recall@k']:.2f}"
                    if m["All-Recall@k"] is not None
                    else "—",
                    "MRR": f"{m['MRR']:.2f}" if m["MRR"] is not None else "—",
                }
            )
        st.table(table_rows)

        # Per-method retrieved passages with ✓/✗ annotations
        st.subheader("Retrieved passages")
        gold_set = set(gold_ids)
        method_cols = st.columns(len(active_methods))
        for col, method in zip(method_cols, active_methods):
            result = results[method]
            with col:
                st.markdown(f"**{method}**")
                if not result.passages:
                    st.caption("No passages retrieved.")
                    continue
                for p in result.passages:
                    marker = "✓" if p.doc_id in gold_set else "✗"
                    color = "green" if p.doc_id in gold_set else "red"
                    st.markdown(f":{color}[{marker}] **{p.title}** `{p.score:.3f}`")
                    st.caption(p.text[:200] + ("…" if len(p.text) > 200 else ""))
                if result.trace.sub_queries or result.trace.rerank_scores or result.trace.notes:
                    with st.expander("Reasoning trace"):
                        render_trace(result.trace, result.passages)
