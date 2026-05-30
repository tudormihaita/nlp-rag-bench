from dataclasses import dataclass
from typing import Iterator

from ragbench.generation.llm import Generator
from ragbench.generation.prompts import NO_RAG_PROMPT, RAG_PROMPT
from ragbench.retrievers.base import RetrievedPassage, RetrievalTrace, Retriever


@dataclass
class PipelineResult:
    answer: str
    passages: list[RetrievedPassage]
    trace: RetrievalTrace


class RAGPipeline:
    def __init__(
        self,
        generator: Generator,
        retriever: Retriever | None = None,
        top_k: int = 5,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.top_k = top_k

    def _retrieve_and_prompt(
        self, question: str
    ) -> tuple[str, list[RetrievedPassage], RetrievalTrace]:
        if self.retriever is None:
            return NO_RAG_PROMPT.format(question=question), [], RetrievalTrace(notes="no-RAG")

        passages, trace = self.retriever.retrieve(question, k=self.top_k)
        context = "\n\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
        prompt = RAG_PROMPT.format(context=context, question=question)
        return prompt, passages, trace

    def run(self, question: str) -> PipelineResult:
        """Blocking call; returns a complete PipelineResult. Used by the evaluation runner."""
        prompt, passages, trace = self._retrieve_and_prompt(question)
        answer = self.generator.generate(prompt)
        return PipelineResult(answer=answer, passages=passages, trace=trace)

    def stream(self, question: str) -> tuple[Iterator[str], list[RetrievedPassage], RetrievalTrace]:
        """
        Streaming call; retrieval is synchronous, generation is streamed.
        Returns (token_iterator, passages, trace) so the UI can show
        retrieved context immediately while the answer streams in below.
        """
        prompt, passages, trace = self._retrieve_and_prompt(question)
        return self.generator.stream(prompt), passages, trace