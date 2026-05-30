import re

import chromadb
from loguru import logger

from ragbench.generation.llm import Generator
from ragbench.generation.prompts import DECOMPOSITION_PROMPT
from ragbench.indexing.embedder import Embedder

from .base import RetrievedPassage, RetrievalTrace
from .naive import NaiveRetriever


class DecompositionRetriever:
    """
    Query decomposition: the LLM splits the question into single-hop sub-questions,
    each is retrieved independently, results are merged with deduplication
    (same doc_id -> keep highest score), top-k returned overall.

    Falls back to the original query (behaves like NaiveRetriever) if the LLM
    returns output that cannot be parsed into sub-questions.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        generator: Generator,
    ) -> None:
        self.naive = NaiveRetriever(collection, embedder)
        self.generator = generator

    def _decompose(self, question: str) -> list[str]:
        """Call the generator model and parse numbered sub-questions. Returns [] on parse failure"""
        raw = self.generator.generate(DECOMPOSITION_PROMPT.format(question=question))
        sub_questions = []
        for line in raw.strip().splitlines():
            # Accept "1. ...", "1) ...", "1: ..."
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

        # Retrieve per sub-question; keep highest-scoring passage per doc_id
        best: dict[str, RetrievedPassage] = {}
        for sq in sub_questions:
            for passage in self.naive.retrieve(sq, k=k)[0]:
                if passage.doc_id not in best or passage.score > best[passage.doc_id].score:
                    best[passage.doc_id] = passage

        merged = sorted(best.values(), key=lambda p: p.score, reverse=True)[:k]
        return merged, RetrievalTrace(sub_queries=sub_questions)
