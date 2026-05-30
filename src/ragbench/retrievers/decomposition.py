import re
from collections import defaultdict
from dataclasses import replace

import chromadb
from loguru import logger

from ragbench.generation.llm import Generator
from ragbench.generation.prompts import DECOMPOSITION_PROMPT, INTERMEDIATE_PROMPT, REWRITE_PROMPT
from ragbench.indexing.embedder import Embedder

from .base import RetrievalTrace, RetrievedPassage
from .naive import NaiveRetriever

_RRF_C = 60  # standard RRF constant

# Markers indicating the LLM failed to produce a useful intermediate answer
_REFUSAL_MARKERS = (
    "i don't know",
    "unknown",
    "cannot determine",
    "not provided",
    "not specified",
    "no information",
    "unclear",
    "n/a",
    "none",
    "the context does not",
    "the passage does not",
    "i'm not sure",
    "not mentioned",
    "not stated",
)


def _is_useful_answer(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 2 or len(t) > 150:
        return False
    return not any(m in t for m in _REFUSAL_MARKERS)


class DecompositionRetriever:
    """
    Query decomposition retriever with two modes:

    Static (iterative=False, default):
        LLM splits the question into self-contained sub-questions. Each is
        retrieved independently and results merged with RRF. Two LLM calls
        total (decompose + generate).

    Iterative (iterative=True):
        After each sub-question retrieval, generates an intermediate answer
        and rewrites the next sub-question to include the resolved entity
        (e.g. "Who is the spouse of the actor?" + "Paul Bettany" →
        "Who is Paul Bettany's spouse?"). This directly targets bridge
        entities in subsequent retrievals. Adds N-1 rewrite calls and N
        intermediate-answer calls per question vs static mode.

    Both modes fall back to the original query if decomposition fails.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        generator: Generator,
        k_per_subquery: int = 10,
        k_for_intermediate: int = 4,
        iterative: bool = False,
    ) -> None:
        self.naive = NaiveRetriever(collection, embedder)
        self.generator = generator
        self.k_per_subquery = k_per_subquery
        # Passages used for intermediate answer generation (subset of k_per_subquery)
        self.k_for_intermediate = k_for_intermediate
        self.iterative = iterative

    def _decompose(self, question: str) -> list[str]:
        """Call the LLM and parse numbered sub-questions. Returns [] on parse/sanity failure."""
        raw = self.generator.generate(DECOMPOSITION_PROMPT.format(question=question))
        sub_questions = []
        for line in raw.strip().splitlines():
            m = re.match(r"^\d+[.):\s]\s*(.+)$", line.strip())
            if m:
                sq = m.group(1).strip()
                if 3 <= len(sq.split()) <= 25:
                    sub_questions.append(sq)
        if not (2 <= len(sub_questions) <= 5):
            logger.warning(
                f"Decomposition produced {len(sub_questions)} sub-questions (expected 2-5); falling back"
            )
            return []
        return sub_questions

    def _rewrite_subquestion(self, sq: str, intermediate_answers: list[str]) -> str:
        """
        Use the LLM to rewrite a sub-question as a self-contained query by
        substituting known intermediate answers for vague references.
        Falls back to simple string enrichment if the rewrite looks malformed.
        """
        facts = "; ".join(intermediate_answers)
        raw = self.generator.generate(REWRITE_PROMPT.format(question=sq, facts=facts))
        rewritten = raw.strip().splitlines()[0].strip()
        if 2 <= len(rewritten.split()) <= 30:
            return rewritten
        # Fallback: append known facts to original sub-question
        return f"{sq} (known: {facts})"

    def _rrf_merge(
        self, per_query_results: list[list[RetrievedPassage]], k: int
    ) -> list[RetrievedPassage]:
        """Merge passage lists with Reciprocal Rank Fusion. Uses replace() to avoid mutation."""
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_objects: dict[str, RetrievedPassage] = {}
        for passages in per_query_results:
            for rank, p in enumerate(passages):
                rrf_scores[p.doc_id] += 1.0 / (_RRF_C + rank + 1)
                if p.doc_id not in doc_objects:
                    doc_objects[p.doc_id] = p
        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [replace(doc_objects[doc_id], score=score) for doc_id, score in top]

    def _retrieve_static(
        self, sub_questions: list[str], k: int
    ) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        """RRF over all sub-questions without intermediate generation."""
        per_query = [self.naive.retrieve(sq, k=self.k_per_subquery)[0] for sq in sub_questions]
        return self._rrf_merge(per_query, k), RetrievalTrace(sub_queries=sub_questions)

    def _retrieve_iterative(
        self, sub_questions: list[str], k: int
    ) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        """
        For each sub-question: optionally rewrite using prior intermediate answers
        to resolve bridge entities, retrieve, generate intermediate answer, repeat.
        """
        intermediate_answers: list[str] = []
        enriched_queries: list[str] = []
        per_query: list[list[RetrievedPassage]] = []

        for sq in sub_questions:
            # Rewrite sub-question using resolved entities from prior hops
            if intermediate_answers:
                enriched = self._rewrite_subquestion(sq, intermediate_answers)
            else:
                enriched = sq
            enriched_queries.append(enriched)

            sq_passages, _ = self.naive.retrieve(enriched, k=self.k_per_subquery)
            per_query.append(sq_passages)

            # Generate intermediate answer from top-k_for_intermediate passages only
            context = "\n\n".join(
                f"[{i + 1}] {p.text}" for i, p in enumerate(sq_passages[: self.k_for_intermediate])
            )
            ans = self.generator.generate(INTERMEDIATE_PROMPT.format(context=context, question=sq))
            if _is_useful_answer(ans):
                intermediate_answers.append(ans.strip())

        trace = RetrievalTrace(
            sub_queries=sub_questions,
            intermediate_answers=intermediate_answers,
            enriched_queries=enriched_queries,
        )
        return self._rrf_merge(per_query, k), trace

    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        sub_questions = self._decompose(query)

        if not sub_questions:
            logger.warning(
                "Decomposition produced no sub-questions; falling back to original query"
            )
            passages, _ = self.naive.retrieve(query, k=k)
            return passages, RetrievalTrace(
                notes="decomposition failed; falling back to original query"
            )

        if self.iterative:
            return self._retrieve_iterative(sub_questions, k)
        return self._retrieve_static(sub_questions, k)
