# Author: Tudor Mihaita
from collections.abc import Sequence

import streamlit as st

from ragbench.retrievers.base import RetrievalTrace, RetrievedPassage


def render_trace(
    trace: RetrievalTrace,
    passages: Sequence[RetrievedPassage] | None = None,
) -> None:
    """Render a retrieval trace inside the current Streamlit container.

    - Re-ranking: shows CrossEncoder scores paired with passage titles.
    - Decomposition (static): numbered sub-questions with explanatory caption.
    - Decomposition (iterative): sub-questions + enriched rewrites + intermediate answers;
      flags answers that look like questions (bridge-entity resolution failed).
    """
    if trace.rerank_scores and passages:
        if trace.notes:
            st.caption(trace.notes)
        st.caption("Cross-encoder scores for retained passages (higher = more relevant):")
        for p, score in zip(passages, trace.rerank_scores):
            filled = int(round(score * 10))
            bar = "█" * filled + "░" * (10 - filled)
            st.caption(f"`{score:.3f}` {bar}  {p.title}")
        return

    if trace.sub_queries:
        is_iterative = bool(trace.enriched_queries)
        if is_iterative:
            st.caption(
                "Iterative decomposition - each sub-question is rewritten using prior "
                "intermediate answers to resolve bridge entities before retrieval."
            )
        else:
            st.caption(
                "Static decomposition - sub-questions are retrieved independently "
                "and merged with Reciprocal Rank Fusion (RRF)."
            )

        for i, sq in enumerate(trace.sub_queries, 1):
            enriched = trace.enriched_queries[i - 1] if i <= len(trace.enriched_queries) else None
            ans = trace.intermediate_answers[i - 1] if i <= len(trace.intermediate_answers) else None

            st.markdown(f"**{i}.** {sq}")
            if enriched and enriched != sq:
                st.caption(f"→ _{enriched}_")
            if ans:
                is_question = ans.strip().endswith("?")
                icon = "⚠️" if is_question else "✦"
                st.caption(f"{icon} {ans}")
                if is_question:
                    st.caption(
                        "↳ _Model produced a question instead of a factual answer - "
                        "the sub-question was used for retrieval without bridge-entity rewriting._"
                    )

    if trace.notes and not trace.rerank_scores:
        st.caption(f"Note: {trace.notes}")
