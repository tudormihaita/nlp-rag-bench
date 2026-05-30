# nlp-rag-bench

#### NLP — Multi-Hop QA RAG Methods Benchmark

Coursework project for NLP. A practical benchmark and interactive application comparing retrieval-augmented generation (RAG) strategies on **open-domain multi-hop question answering**, using the MuSiQue dataset. The central thesis: **classic RAG handles single-hop reasoning well but degrades sharply as questions require more reasoning steps; training-free retrieval-side improvements can recover much of that gap without modifying the generator**.

---

## Task: Open-Domain Multi-Hop Question Answering

Given a question $q$ and a passage corpus $\mathcal{C} = \{p_1, \ldots, p_N\}$, the system must produce an answer string $a$. Solving the task requires (1) identifying a set of supporting passages $\mathcal{S} \subseteq \mathcal{C}$, and (2) synthesizing $a$ from $\mathcal{S}$. In multi-hop QA, $|\mathcal{S}| \geq 2$ and the passages must be combined via reasoning (e.g., *"Who is the spouse of the performer of Imagine?"* → identify performer in one passage, find their spouse in another).

This sits at the intersection of three NLP sub-tasks:
- **Information Retrieval** — ranking passages by relevance to the query
- **Reading Comprehension / Answer Generation** — producing an answer from retrieved context
- **Multi-Hop Reasoning** — combining facts across multiple passages

The project's contribution is a **comparative evaluation of retrieval strategies within the RAG paradigm**, with results stratified by reasoning depth (2/3/4 hops) to expose how different methods scale with question difficulty.

---

## Dataset: MuSiQue

[MuSiQue](https://github.com/StonyBrookNLP/musique) (Trivedi et al., 2022) is a multi-hop QA benchmark constructed by composing verified single-hop questions into 2-, 3-, and 4-hop chains. Unlike HotpotQA, MuSiQue actively filters out shortcut-solvable questions, making it a stronger benchmark for distinguishing RAG methods.

### Structure (MuSiQue-Answerable, dev split)

Each question is one JSON object:

```json
{
  "id": "2hop__123_456",
  "question": "Who is the spouse of the performer of Imagine?",
  "answer": "Yoko Ono",
  "answer_aliases": ["Ono Yoko"],
  "question_decomposition": [
    {"id": 0, "question": "Who performed Imagine?", "answer": "John Lennon", "paragraph_support_idx": 3},
    {"id": 1, "question": "Who is the spouse of #1 ?", "answer": "Yoko Ono", "paragraph_support_idx": 17}
  ],
  "paragraphs": [
    {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": false},
    ...
    {"idx": 3, "title": "John Lennon", "paragraph_text": "...", "is_supporting": true}
  ]
}
```

Key fields:
- `question`, `answer`, `answer_aliases` — input and gold output
- `paragraphs[].is_supporting` — flags gold passages (used for retrieval metrics)
- `id` prefix (`2hop__`, `3hop1__`, `4hop1__`, …) — encodes hop count for stratification
- `question_decomposition` — gold reasoning chain (used for analysis only, never for retrieval)

### Sampling strategy

We use the **dev split** (test labels are not public) and sample **500 questions stratified by hop count**: 200 × 2-hop, 200 × 3-hop, 100 × 4-hop. This is enough for stable metrics and runs end-to-end in a few hours on a laptop.

### Corpus construction

MuSiQue ships ~20 candidate paragraphs per question (2–4 gold + ~16 distractors). We build a **pooled corpus**: deduplicated union of all paragraphs across all sampled questions (~10k unique passages). This makes retrieval competitive — gold passages must compete against distractors from *other* questions, not just their own.

No additional chunking is needed — MuSiQue passages are already paragraph-sized.

---

## RAG Methods Compared

We compare **3 RAG methods plus a no-RAG baseline (four total)**. All methods are **training-free retrieval-side techniques**: they share the same dense index, the same generator, and the same prompt template — only the retrieval orchestration changes. This isolates the retrieval variable for a fair comparison.

| # | Method | Description | Reference |
|---|--------|-------------|-----------|
| 1 | **No-RAG baseline** | Bare LLM call, empty context | Quantifies parametric memory; lower bound |
| 2 | **Classic RAG** | Top-k dense retrieval (cosine similarity) | Lewis et al. (2020) |
| 3 | **Cross-encoder re-ranking** | Retrieve top-30 with dense, re-rank with cross-encoder, keep top-5 | Nogueira & Cho (2019) |
| 4 | **Query decomposition** | LLM splits the question into single-hop sub-questions, retrieves per sub-question, merges results | Trivedi et al. (2023) — IRCoT |

**Expected story** in the final results table: classic RAG performs reasonably on 2-hop, collapses on 3/4-hop. Re-ranking gives uniform gains across hop counts. Decomposition gives *increasing* gains with hop count — that's the visual punchline of the project.

**Note on training:** none of these methods require training any model. The cross-encoder is loaded pre-trained from HuggingFace; the decomposer is the same LLM used as the generator, prompted differently. Everything is inference + evaluation.

---

## Generator Model Choice

We use a **small open-weight LLM via Ollama** rather than a frontier API model. Reason: frontier models (GPT-4, Claude) have memorized large portions of Wikipedia, so the no-RAG baseline scores artificially high on MuSiQue and the gaps between RAG methods compress. A 3–8B open model has weaker parametric memory, producing cleaner, more visible RAG gains.

**Default:** `llama3.1:8b-instruct-q4_K_M` (~5GB quantized, runs on most laptops).
**Alternatives:** `llama3.2:3b`, `qwen2.5:3b-instruct`, `phi3:mini` — faster, lower quality, fine for iteration.

**Methodology check:** evaluate the chosen LLM with no retrieval on 50 random MuSiQue questions. If F1 > 20%, the model is memorizing too much — drop to a smaller variant.

---

## Embedding Model Choice

The embedder is fixed across all retrieval methods and used both at indexing time and at query time. The default is **`BAAI/bge-small-en-v1.5`** — strong MTEB scores on retrieval, small (~133MB, 384 dim), and well-suited to Wikipedia-style passages.

| Model | Dim | Size | Notes |
|---|---|---|---|
| **`BAAI/bge-small-en-v1.5`** ✓ default | 384 | ~133MB | Fast, strong on retrieval, well-balanced |
| `BAAI/bge-base-en-v1.5` | 768 | ~440MB | Stronger quality if compute allows; ~2× slower |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90MB | Very fast, slightly weaker than BGE; widely used fallback |
| `intfloat/e5-base-v2` | 768 | ~440MB | Strong retrieval scores; **requires `"query: "` / `"passage: "` prefixes** |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~550MB | Recent, strong; **requires `"search_query: "` / `"search_document: "` prefixes** |

**Important:** the embedder choice is bound to the index. Different models produce different vectors with different dimensions, so swapping the embedder requires rebuilding the index. We persist each index under a model-specific folder (`chroma_db/<model_slug>/`) so multiple indexes can coexist.

---

## Evaluation Metrics

All metrics computed per question, then aggregated by `(method, hop_count)` slices.

### Retrieval (using `is_supporting` paragraphs as ground truth)

| Metric | Definition |
|---|---|
| **Hit@k** | Did *any* gold paragraph appear in the top-k? |
| **Recall@k** | Fraction of gold paragraphs retrieved in top-k |
| **All-Recall@k** | Were *all* gold paragraphs retrieved? (harsh metric, exposes multi-hop failures) |
| **MRR** | Mean reciprocal rank of the first gold paragraph |

### Generation (using `answer` + `answer_aliases` as references)

| Metric | Definition |
|---|---|
| **Exact Match (EM)** | Strict match after SQuAD-style normalization (lowercase, strip articles/punct) |
| **Token F1** | Token-level overlap with gold answer (the standard QA metric) |
| **LLM-as-judge** | Binary correctness scored by Claude / GPT-4o (catches semantically correct answers EM/F1 miss) |

### Optional (if time permits): RAGAS

Reference-free LLM-as-judge metrics — **faithfulness**, **context precision**, **context recall**. Useful but adds API cost and noise; not required for the core results.

### Headline output

A single comparison table:

| Method | Recall@5 (2h) | Recall@5 (3h) | Recall@5 (4h) | F1 (2h) | F1 (3h) | F1 (4h) |
|---|---|---|---|---|---|---|
| No-RAG | — | — | — | … | … | … |
| Classic RAG | … | … | … | … | … | … |
| Re-ranking | … | … | … | … | … | … |
| Decomposition | … | … | … | … | … | … |

---

## Application

A Streamlit app with three modes:

### 1. Chat Mode (primary UX)
- Free-form question input, conversational layout
- Sidebar: method selector, generator selector, top-k slider
- Each answer comes with an expandable **"Retrieved context"** panel showing the chunks used, their scores, and source paragraph titles
- For query decomposition: a **"Reasoning trace"** panel showing the sub-questions and what was retrieved for each — strong visual differentiator

### 2. Compare Mode
- One question, all methods run in parallel
- Side-by-side answer columns with retrieval traces
- The signature screenshot for the project: same question, no-RAG hallucinates, classic RAG gets close, decomposition gets it right

### 3. Benchmark Mode
- Pull a random question from the sampled MuSiQue set (filterable by hop count)
- Reveal toggles for the gold answer and gold supporting paragraphs
- Run all methods, display answers and per-question metrics live with ✓/✗ annotations for retrieved-vs-gold passages
- Lets a user *experience* the difficulty curve, not just read it in the report

### Optional deployment
Local Streamlit is sufficient for the demo. If public access is required, push to **HuggingFace Spaces** with a hosted LLM endpoint (Together AI, Anthropic, or OpenAI) instead of local Ollama.

---

## Technology Stack

| Component | Choice | Notes |
|---|---|---|
| **Orchestration** | LlamaIndex `llama-index-core` | Clean retriever-swapping API; can drop to raw ChromaDB if abstraction fights back |
| **Vector store** | ChromaDB (persistent) | In-process, no server needed |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | Swappable via config; see table above |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Runs on CPU |
| **Generator LLM** | Ollama (`llama3.1:8b-instruct`, fallback `llama3.2:3b`, `qwen2.5:3b`) | Local inference, no API key needed |
| **LLM-as-judge** | Anthropic Claude or OpenAI GPT-4o | Evaluation only; called via API |
| **UI** | Streamlit | Three-mode layout (chat / compare / benchmark) |
| **Evaluation** | Custom (EM, F1, Recall@k, MRR) + optional RAGAS | SQuAD-style normalization for EM/F1 |
| **Dataset** | MuSiQue-Answerable v1.0 (dev split) | Sampled 500 questions stratified by hop count |
| **Package manager** | `uv` | |

---

## Project Structure

```
nlp-rag-bench/
├── pyproject.toml
├── .env.example                # API keys template (LLM-as-judge)
├── data/
│   ├── musique/
│   │   └── musique_ans_v1.0_dev.jsonl
│   └── processed/
│       ├── sampled_questions.json    # 500 stratified questions
│       └── pooled_corpus.json        # ~10k deduplicated paragraphs
├── chroma_db/                  # persisted vector stores, one per embedder (gitignored)
│   ├── bge-small-en-v1.5/
│   └── bge-base-en-v1.5/
├── src/
│   └── ragbench/
│       ├── __init__.py
│       ├── config.py           # pydantic-settings: paths, model names, top-k, sampling seed
│       ├── data/
│       │   ├── loader.py       # parse MuSiQue JSONL, extract hop counts from IDs
│       │   ├── sampler.py      # stratified sampling by hop count
│       │   └── corpus.py       # pool + deduplicate paragraphs across questions
│       ├── indexing/
│       │   ├── embedder.py     # wraps sentence-transformers; swappable model
│       │   └── builder.py      # encode corpus + persist to ChromaDB
│       ├── retrievers/
│       │   ├── base.py         # Retriever protocol + RetrievedPassage + RetrievalTrace
│       │   ├── naive.py        # classic dense top-k
│       │   ├── reranking.py    # dense top-30 → cross-encoder rerank → top-5
│       │   └── decomposition.py# LLM decomposes → retrieve per sub-question → merge
│       ├── generation/
│       │   ├── llm.py          # Ollama wrapper
│       │   └── prompts.py      # shared QA prompt + decomposition prompt
│       ├── pipeline.py         # RAGPipeline: composes retriever + generator
│       ├── evaluation/
│       │   ├── metrics.py      # EM, F1, Recall@k, MRR, All-Recall@k
│       │   ├── judge.py        # optional LLM-as-judge
│       │   └── runner.py       # evaluate one method over the sampled set
│       └── app/
│           ├── streamlit_app.py    # entrypoint with mode selector
│           ├── chat_mode.py
│           ├── compare_mode.py
│           └── benchmark_mode.py
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_results_analysis.ipynb
└── scripts/
    ├── prepare_data.py         # CLI: load → sample → pool → save processed/
    ├── build_index.py          # CLI: embed pooled corpus → persist ChromaDB
    ├── evaluate.py             # CLI: run all methods, save results, print table
    └── memorization_check.py   # CLI: no-RAG baseline on 50 questions
```

---

## Pipeline / Process

The pipeline runs once for setup, then is reused for every query and every evaluation run.

### Setup (one-time)

```
[MuSiQue dev JSONL]
        │
        ▼  prepare_data.py
[sample 500 questions by hop count]
        │
        ▼
[pool & dedupe paragraphs] ──► data/processed/
        │
        ▼  build_index.py
[embed all paragraphs with chosen embedder]
        │
        ▼
[persist to ChromaDB] ──► chroma_db/<model_slug>/
```

### Per query (runtime, swappable)

```
       [user question]
              │
              ▼
   ┌─────────────────────┐
   │   Retriever (one of):│
   │   • naive            │
   │   • reranking        │
   │   • decomposition    │
   └─────────────────────┘
              │
              ▼ top-k passages
   [build prompt: question + context]
              │
              ▼
   [Llama 3.1 8B via Ollama]
              │
              ▼
       [generated answer]
              │
              ▼
   [render in UI: answer + retrieved chunks + reasoning trace]
```

### Per evaluation run (offline)

```
for method in [no_rag, naive, reranking, decomposition]:
    for question in sampled_500:
        retrieved = method.retrieve(question)
        answer    = generator.generate(question, retrieved)
        record(retrieval_metrics(retrieved, gold_paragraphs),
               generation_metrics(answer, gold_answers),
               hop_count(question.id))

aggregate by (method, hops) → print headline table
```

The retriever is the only swappable component. Generator, embedder, prompt, and corpus are fixed — that's what makes the comparison fair.

---

## Implementation Plan

Suggested split: **Person A** owns data + indexing + classic RAG + UI scaffolding. **Person B** owns the two improvement methods + evaluation + UI mode-specific panels.

---

### Phase 1 — Data, Embedding & Indexing

**Files involved**

| File | Responsibility |
|---|---|
| `data/loader.py` | Read `musique_ans_v1.0_dev.jsonl`, parse each record, extract hop count from `id` prefix |
| `data/sampler.py` | Stratified sampling by hop count (200 × 2-hop, 200 × 3-hop, 100 × 4-hop), fixed seed |
| `data/corpus.py` | Pool + dedupe paragraphs across sampled questions; build `gold_paragraphs[qid] -> set` mapping |
| `indexing/embedder.py` | Thin wrapper around `sentence-transformers`; reads model name from config |
| `indexing/builder.py` | Encode pooled corpus in batches, persist to ChromaDB under model-specific path |
| `scripts/prepare_data.py` | CLI orchestrating loader → sampler → corpus, writes `data/processed/` |
| `scripts/build_index.py` | CLI orchestrating embedder → builder, writes `chroma_db/<model_slug>/` |

**Embedder design** — single wrapper used by both indexer and retrievers (they must match):

```python
# src/ragbench/indexing/embedder.py
from sentence_transformers import SentenceTransformer

# Models that require input prefixes for query vs passage.
PREFIXED_MODELS = {
    "intfloat/e5-base-v2":            {"query": "query: ",        "passage": "passage: "},
    "intfloat/e5-small-v2":           {"query": "query: ",        "passage": "passage: "},
    "nomic-ai/nomic-embed-text-v1.5": {"query": "search_query: ", "passage": "search_document: "},
}

class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.prefixes = PREFIXED_MODELS.get(model_name, {"query": "", "passage": ""})

    def encode_passages(self, texts: list[str], batch_size: int = 64):
        prefixed = [self.prefixes["passage"] + t for t in texts]
        return self.model.encode(prefixed, batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True)

    def encode_query(self, text: str):
        return self.model.encode(self.prefixes["query"] + text,
                                 normalize_embeddings=True)

    @property
    def slug(self) -> str:
        return self.model_name.split("/")[-1]  # used in index path
```

**Index builder** — persists per-model so swapping embedders doesn't clobber:

```python
# src/ragbench/indexing/builder.py
import chromadb
from .embedder import Embedder

def build_index(passages: list[dict], embedder: Embedder, base_path: str = "chroma_db"):
    path = f"{base_path}/{embedder.slug}"
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name="musique", metadata={"hnsw:space": "cosine"}
    )
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode_passages(texts)
    collection.add(
        ids=[p["doc_id"] for p in passages],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"title": p["title"]} for p in passages],
    )
    return collection
```

**Swapping the embedding model** — change one line in `config.py`, re-run `build_index.py`:

```python
# src/ragbench/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # ← change here
    generator_model:  str = "llama3.1:8b-instruct-q4_K_M"
    top_k: int = 5
    sampling_seed: int = 42
    # ... no other code changes needed
```

**Checklist**
- [A] Implement loader + sampler + corpus pooler
- [A] Implement `Embedder` and `build_index`
- [A] Run `prepare_data.py` then `build_index.py`
- [A] `memorization_check.py` — verify no-RAG F1 < 20% on the chosen LLM
- **Milestone:** corpus pooled, index persisted, sampled questions loaded; chosen LLM verified non-memorizing

---

### Phase 2 — Generator, Retriever Interface & Pipeline

This phase establishes the abstractions that the two improvement methods will plug into. Get this right and Phase 3 is mostly filling in two small classes.

**Files involved**

| File | Responsibility |
|---|---|
| `generation/llm.py` | Wrap Ollama HTTP API behind a simple `Generator` interface |
| `generation/prompts.py` | Centralized prompt templates (RAG, no-RAG, decomposition) |
| `retrievers/base.py` | `Retriever` Protocol, `RetrievedPassage`, `RetrievalTrace` dataclasses |
| `retrievers/naive.py` | Classic dense top-k retriever |
| `pipeline.py` | `RAGPipeline` — composes a retriever + generator; supports no-RAG via `retriever=None` |
| `app/chat_mode.py` | Streamlit chat UI with method selector wired to one pipeline-per-method |

**Generator wrapper** — one class, one method, shared across all RAG methods:

```python
# src/ragbench/generation/llm.py
import ollama

class Generator:
    def __init__(self, model: str = "llama3.1:8b-instruct-q4_K_M", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"].strip()
```

**Prompt templates** — kept identical across RAG methods to isolate the retrieval variable:

```python
# src/ragbench/generation/prompts.py
RAG_PROMPT = """Answer the question based ONLY on the context below. \
If the answer cannot be determined from the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

NO_RAG_PROMPT = """Answer the question concisely. If you don't know, say "I don't know."

Question: {question}

Answer:"""

DECOMPOSITION_PROMPT = """Decompose the following multi-hop question into 2-4 \
simple sub-questions that can be answered one at a time. Output ONLY the sub-questions, \
one per line, numbered.

Question: {question}

Sub-questions:"""
```

**Retriever protocol** — the contract every method implements:

```python
# src/ragbench/retrievers/base.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    title: str
    score: float

@dataclass
class RetrievalTrace:
    """Optional metadata the UI can render (sub-questions, rerank scores, etc.)."""
    sub_queries: list[str] = field(default_factory=list)
    rerank_scores: list[float] = field(default_factory=list)
    notes: str = ""

class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> tuple[list[RetrievedPassage], RetrievalTrace]:
        ...
```

**Naive retriever** — the reference implementation:

```python
# src/ragbench/retrievers/naive.py
from .base import RetrievedPassage, RetrievalTrace

class NaiveRetriever:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.embedder.encode_query(query)
        res = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=k)
        passages = [
            RetrievedPassage(doc_id=i, text=t, title=m["title"], score=1.0 - d)
            for i, t, m, d in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0]
            )
        ]
        return passages, RetrievalTrace()
```

**Pipeline** — composes a retriever and generator; handles no-RAG by passing `retriever=None`:

```python
# src/ragbench/pipeline.py
from dataclasses import dataclass
from .retrievers.base import Retriever, RetrievedPassage, RetrievalTrace
from .generation.llm import Generator
from .generation.prompts import RAG_PROMPT, NO_RAG_PROMPT

@dataclass
class PipelineResult:
    answer: str
    passages: list[RetrievedPassage]
    trace: RetrievalTrace

class RAGPipeline:
    def __init__(self, generator: Generator, retriever: Retriever | None = None, top_k: int = 5):
        self.generator = generator
        self.retriever = retriever
        self.top_k = top_k

    def run(self, question: str) -> PipelineResult:
        if self.retriever is None:
            answer = self.generator.generate(NO_RAG_PROMPT.format(question=question))
            return PipelineResult(answer=answer, passages=[], trace=RetrievalTrace(notes="no-RAG"))

        passages, trace = self.retriever.retrieve(question, k=self.top_k)
        context = "\n\n".join(f"[{i+1}] {p.text}" for i, p in enumerate(passages))
        prompt = RAG_PROMPT.format(context=context, question=question)
        answer = self.generator.generate(prompt)
        return PipelineResult(answer=answer, passages=passages, trace=trace)
```

**How the UI uses it** — one pipeline instance per method, sidebar swaps which is active:

```python
# src/ragbench/app/chat_mode.py (sketch)
PIPELINES = {
    "No-RAG":        RAGPipeline(generator=gen, retriever=None),
    "Classic RAG":   RAGPipeline(generator=gen, retriever=NaiveRetriever(collection, embedder)),
    "Re-ranking":    RAGPipeline(generator=gen, retriever=ReRankRetriever(...)),     # Phase 3
    "Decomposition": RAGPipeline(generator=gen, retriever=DecompositionRetriever(...)),# Phase 3
}

method = st.sidebar.selectbox("Retrieval method", list(PIPELINES.keys()))
question = st.chat_input("Ask a question")
if question:
    result = PIPELINES[method].run(question)
    st.write(result.answer)
    with st.expander("Retrieved context"):
        for p in result.passages:
            st.markdown(f"**{p.title}** (score: {p.score:.3f})\n\n{p.text}")
    if result.trace.sub_queries:
        with st.expander("Reasoning trace"):
            for sq in result.trace.sub_queries:
                st.write(f"• {sq}")
```

**Checklist**
- [A] Implement `Generator`, `Retriever` protocol, `NaiveRetriever`, `RAGPipeline`
- [A] Write `prompts.py` with shared templates
- [A] Wire chat mode with method selector — only `No-RAG` and `Classic RAG` active for now
- **Milestone:** working chat UI; switching between No-RAG and Classic RAG produces visibly different answers on a hard MuSiQue question

---

### Phase 3 — Improvement Methods

Both methods implement the same `Retriever` protocol and plug into the existing `RAGPipeline` without modification.

- [B] **Cross-encoder re-ranking** (~30 lines) — `retrievers/reranking.py`
  - Wrap `NaiveRetriever` to fetch top-30; score each `(query, passage)` pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`; return top-5
  - Trace exposes rerank scores
- [B] **Query decomposition** (~80–100 lines) — `retrievers/decomposition.py`
  - Prompt the LLM with `DECOMPOSITION_PROMPT` to produce sub-questions
  - Parse sub-questions, retrieve top-k per sub-question via `NaiveRetriever`
  - Merge with deduplication (highest score wins), return top-k overall
  - Trace exposes the sub-questions for the UI's "Reasoning trace" panel
- [B] Register both pipelines in the chat-mode method selector

**Milestone:** all 4 methods (No-RAG + Classic + Re-ranking + Decomposition) selectable; per-question latency reasonable (decomposition will be slowest, ~10–20s per question with 8B model).

---

### Phase 4 — Evaluation

- [B] Implement EM, F1, Recall@k, MRR, All-Recall@k in `evaluation/metrics.py` (use SQuAD normalization for EM/F1)
- [B] `runner.py` iterates `(method, question)`, calls `pipeline.run()`, records metrics + hop count
- [B] `evaluate.py` saves results to CSV; print aggregate table grouped by `(method, hops)`
- **Milestone:** headline comparison table generated; CSV saved for the results dashboard

---

### Phase 5 — Compare & Benchmark Modes + Polish

- [A] Compare mode: same query → all 4 methods side-by-side
- [B] Benchmark mode: random MuSiQue question, gold reveal toggle, live metrics per method
- [A] Reasoning-trace panel for decomposition; rerank-score panel for re-ranking
- [A] Retrieved-chunks panel with scores + source titles + ✓/✗ gold annotations (benchmark mode only)

---

### Phase 6 — Report & Demo

- Write report: task formulation, methods + citations, results table, hop-count analysis, discussion of when each method helps
- Slides with the headline screenshot (compare mode: failure → success progression)
- Optional: deploy to HuggingFace Spaces with hosted LLM

---

## Repository Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for package management
- [Ollama](https://ollama.com/) for local LLM inference
- ~10GB disk space (corpus + ChromaDB + LLM weights)

### Initial setup

```bash
# Clone and enter the repo
git clone <repo-url> nlp-rag-bench && cd nlp-rag-bench

# Install dependencies
uv sync

# Install with dev extras (notebooks, formatting)
uv sync --extra dev

# Copy and fill in API keys (only needed for LLM-as-judge)
cp .env.example .env

# Pull the generator LLM
ollama pull llama3.1:8b-instruct-q4_K_M

# Download MuSiQue dev split (see github.com/StonyBrookNLP/musique)
# Place at: data/musique/musique_ans_v1.0_dev.jsonl
```

### Running the pipeline

```bash
# 1. Prepare data (load + sample + pool corpus)
uv run python scripts/prepare_data.py

# 2. Build the vector index (under chroma_db/<embedder_slug>/)
uv run python scripts/build_index.py

# 3. Sanity check: confirm the LLM doesn't memorize MuSiQue answers
uv run python scripts/memorization_check.py

# 4. Run full evaluation across all methods
uv run python scripts/evaluate.py

# 5. Launch the application
uv run streamlit run src/ragbench/app/streamlit_app.py
```

### Switching the embedding model

Edit `embedding_model` in `src/ragbench/config.py` (or set the `RAGBENCH_EMBEDDING_MODEL` env var), then rerun `build_index.py`. Each model gets its own directory under `chroma_db/`, so multiple indexes can coexist.

### Switching the retrieval method at runtime

The application's sidebar exposes a method dropdown — selecting a different value reuses the same index and generator and only swaps the retriever. No re-indexing, no model reload.

```python
from ragbench.retrievers import NaiveRetriever, ReRankRetriever, DecompositionRetriever
from ragbench.pipeline import RAGPipeline

pipeline = RAGPipeline(generator=gen, retriever=ReRankRetriever(...))
result = pipeline.run("Who is the spouse of the performer of Imagine?")
```

---

## References

- **Dataset:** Trivedi, Balasubramanian, Khot, Sabharwal (2022). *"MuSiQue: Multihop Questions via Single-hop Question Composition."* TACL. [arXiv:2108.00573](https://arxiv.org/abs/2108.00573)
- **RAG:** Lewis et al. (2020). *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **Re-ranking:** Nogueira & Cho (2019). *"Passage Re-ranking with BERT."* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085)
- **Decomposition (IRCoT):** Trivedi et al. (2023). *"Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions."* ACL 2023. [arXiv:2212.10509](https://arxiv.org/abs/2212.10509)
- **Bi-encoder framework:** Reimers & Gurevych (2019). *"Sentence-BERT."* EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- **Embeddings:** BGE (Beijing Academy of AI); MS MARCO MiniLM cross-encoder