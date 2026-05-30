import re
from collections import defaultdict

import chromadb
from loguru import logger

from ragbench.generation.llm import Generator
from ragbench.generation.prompts import DECOMPOSITION_PROMPT
from ragbench.indexing.embedder import Embedder

from .base import RetrievedPassage, RetrievalTrace
from .naive import NaiveRetriever

_RRF_C = 60  # standard RRF constant; dampens the impact of top-1 rank


class DecompositionRetriever:
    """
    Query decomposition retriever using Reciprocal Rank Fusion (RRF).

    The LLM splits the question into self-contained sub-questions. Each is
    retrieved independently and results are merged with RRF, which is
    rank-based and therefore comparable across sub-queries (unlike raw cosine
    scores, which vary in scale depending on query specificity).

    Falls back to the original query if the LLM output cannot be parsed.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        generator: Generator,
        k_per_subquery: int = 10,
    ) -> None:
        self.naive = NaiveRetriever(collection, embedder)
        self.generator = generator
        self.k_per_subquery = k_per_subquery

    def _decompose(self, question: str) -> list[str]:
        """Call the generator model and parse numbered sub-questions. Returns [] on parse failure."""
        raw = self.generator.generate(DECOMPOSITION_PROMPT.format(question=question))
        sub_questions = []
        for line in raw.strip().splitlines():
            m = re.match(r"^\d+[.):\s]\s*(.+)$", line.strip())
            if m:
                sub_questions.append(m.group(1).strip())
        return sub_questions

    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        sub_questions = self._decompose(query)

        if not sub_questions:
            logger.warning("Decomposition produced no sub-questions; falling back to original query")
            passages, _ = self.naive.retrieve(query, k=k)
            return passages, RetrievalTrace(notes="decomposition failed; falling back to original query")

        # RRF: accumulate 1/(rank + c) across all sub-queries; rank-based so
        # scores are comparable regardless of per-query cosine score scale
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_objects: dict[str, RetrievedPassage] = {}

        for sq in sub_questions:
            for rank, passage in enumerate(self.naive.retrieve(sq, k=self.k_per_subquery)[0]):
                rrf_scores[passage.doc_id] += 1.0 / (_RRF_C + rank + 1)
                if passage.doc_id not in doc_objects:
                    doc_objects[passage.doc_id] = passage

        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = []
        for doc_id, score in top:
            p = doc_objects[doc_id]
            p.score = score
            merged.append(p)

        return merged, RetrievalTrace(sub_queries=sub_questions)
