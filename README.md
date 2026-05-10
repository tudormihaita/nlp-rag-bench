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

We're starting with **4 methods + the no-RAG baseline**; one of methods 3–5 will likely be dropped depending on implementation time. All methods are **training-free retrieval-side techniques**: they share the same dense index, the same generator, and the same prompt template — only the retrieval orchestration changes. This isolates the retrieval variable for a fair comparison.

| # | Method | Description | Why include |
|---|--------|-------------|-------------|
| 1 | **No-RAG baseline** | Bare LLM call, empty context | Quantifies the LLM's parametric memory; lower bound |
| 2 | **Classic RAG** | Top-k dense retrieval (cosine similarity) | Standard reference baseline |
| 3 | **Cross-encoder re-ranking** | Retrieve top-30 with dense, re-rank with cross-encoder, keep top-5 | Post-retrieval refinement; uniform gains across hops |
| 4 | **Multi-query rewriting** | LLM generates N query variants, retrieve for each, merge with Reciprocal Rank Fusion | Diversifies retrieval; helps with under-specified queries |
| 5 | **Query decomposition** | LLM splits the question into single-hop sub-questions, retrieves per sub-question | Directly addresses multi-hop failure mode of classic RAG |

**Expected story** in the final results table: classic RAG performs reasonably on 2-hop, collapses on 3/4-hop. Re-ranking gives uniform gains. Decomposition / multi-query give *increasing* gains with hop count — that's the visual punchline of the project.

---

## Generator Model Choice

We use a **small open-weight LLM via Ollama** rather than a frontier API model. Reason: frontier models (GPT-4, Claude) have memorized large portions of Wikipedia, so the no-RAG baseline scores artificially high on MuSiQue and the gaps between RAG methods compress. A 3–8B open model has weaker parametric memory, producing cleaner, more visible RAG gains.

**Default:** `llama3.1:8b-instruct-q4_K_M` (~5GB quantized, runs on most laptops).
**Alternatives:** `llama3.2:3b`, `qwen2.5:3b-instruct`, `phi3:mini` — faster, lower quality, fine for iteration.

**Methodology check:** evaluate the chosen LLM with no retrieval on 50 random MuSiQue questions. If F1 > 20%, the model is memorizing too much — drop to a smaller variant.

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
| Multi-query | … | … | … | … | … | … |
| Decomposition | … | … | … | … | … | … |

---

## Application

A Streamlit app with three modes:

### 1. Chat Mode (primary UX)
- Free-form question input, conversational layout
- Sidebar: method selector, generator selector, top-k slider
- Each answer comes with an expandable **"Retrieved context"** panel showing the chunks used, their scores, and source paragraph titles
- For query decomposition: a **"Reasoning trace"** panel showing the sub-questions and what was retrieved for each — this is a strong differentiator visually

### 2. Compare Mode
- One question, all methods run in parallel
- Side-by-side answer columns with retrieval traces
- The signature screenshot for the project: same question, the no-RAG column hallucinates, classic RAG gets close, decomposition gets it right

### 3. Benchmark Mode
- Pull a random question from the sampled MuSiQue set (filterable by hop count)
- Show the gold answer and gold supporting paragraphs
- Run all methods, display answers and per-question metrics live
- Lets a user *experience* the difficulty curve, not just read it in the report

### Optional deployment
Local Streamlit is sufficient for the demo. If public access is required, push to **HuggingFace Spaces** with a hosted LLM endpoint (Together AI, Anthropic, or OpenAI) instead of local Ollama.

---

## Technology Stack

| Component | Choice | Notes |
|---|---|---|
| **Orchestration** | LlamaIndex `llama-index-core` | Clean retriever-swapping API; can drop to raw ChromaDB if abstraction fights back |
| **Vector store** | ChromaDB (persistent) | In-process, no server needed |
| **Embeddings** | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | Fixed across all methods; swap to `bge-base` for stronger quality if compute allows |
| **BM25 (for hybrid, if added)** | `rank-bm25` | Separate in-memory index over raw text |
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
ragscope/
├── pyproject.toml
├── .env.example                # API keys template (LLM-as-judge)
├── data/
│   ├── musique/
│   │   └── musique_ans_v1.0_dev.jsonl
│   └── processed/
│       ├── sampled_questions.json    # 500 stratified questions
│       └── pooled_corpus.json        # ~10k deduplicated paragraphs
├── chroma_db/                  # persisted vector store (gitignored)
├── src/
│   └── ragscope/
│       ├── __init__.py
│       ├── config.py           # pydantic-settings: paths, model names, top-k, sampling seed
│       ├── data/
│       │   ├── loader.py       # parse MuSiQue JSONL, extract hop counts from IDs
│       │   ├── sampler.py      # stratified sampling by hop count
│       │   └── corpus.py       # pool + deduplicate paragraphs across questions
│       ├── indexing/
│       │   └── builder.py      # embed corpus, build ChromaDB collection
│       ├── retrievers/
│       │   ├── base.py         # Retriever protocol: retrieve(query, k) -> list[Passage]
│       │   ├── naive.py        # classic dense top-k
│       │   ├── reranking.py    # dense top-30 → cross-encoder rerank → top-5
│       │   ├── multi_query.py  # LLM rewrites → retrieve each → RRF merge
│       │   └── decomposition.py# LLM decomposes → retrieve per sub-question → concat
│       ├── generation/
│       │   ├── llm.py          # Ollama wrapper, prompt template
│       │   └── prompts.py      # shared prompt for all RAG methods + decomposition prompt
│       ├── pipeline.py         # wires retriever + generator; the swap-point
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
[embed all paragraphs with bge-small]
        │
        ▼
[persist to ChromaDB] ──► chroma_db/
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
   │   • multi_query      │
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
for method in [no_rag, naive, reranking, multi_query, decomposition]:
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

Suggested split: **Person A** owns data + indexing + classic RAG + UI scaffolding. **Person B** owns the three improvement methods + evaluation + UI mode-specific panels.

### Phase 1 — Data & Indexing
- [A] Download MuSiQue dev split, write loader and stratified sampler
- [A] Pool and deduplicate corpus, persist to `data/processed/`
- [A] Embed corpus with `bge-small-en-v1.5`, persist to ChromaDB
- [A] `memorization_check.py` — verify no-RAG F1 < 20% on the chosen LLM
- **Milestone:** classic RAG running end-to-end via CLI

### Phase 2 — Generator + Pipeline + UI Scaffold
- [A] Ollama wrapper, shared prompt template
- [A] `Retriever` protocol + `pipeline.py` orchestration
- [A] Streamlit chat mode with method selector (only `naive` wired in for now)
- **Milestone:** working chat UI calling classic RAG

### Phase 3 — Improvement Methods
- [B] Cross-encoder re-ranking (~30 lines) — dense top-30 → rerank → top-5
- [B] Multi-query rewriting (~50 lines) — LLM prompt for N variants + RRF merge
- [B] Query decomposition (~80–100 lines) — LLM prompt for sub-questions + per-hop retrieval
- [B] Each method exposed in the UI's method selector as it lands
- **Milestone:** all 4 methods (+ no-RAG) selectable in chat mode

### Phase 4 — Evaluation
- [B] EM, F1, Recall@k, MRR, All-Recall@k in `evaluation/metrics.py`
- [B] `evaluate.py` runs all methods × all questions, saves CSV + prints aggregate table
- [B] Decision point: is one method underperforming or too slow? Drop it, keep 3.
- **Milestone:** headline comparison table generated

### Phase 5 — Compare & Benchmark Modes + Polish
- [A] Compare mode: same query → all methods side-by-side
- [B] Benchmark mode: random sample from MuSiQue, gold reveal, live metrics
- [A] Reasoning trace panel for decomposition method
- [A] Retrieved chunks panel with scores + source titles

### Phase 6 — Report & Demo
- Write report: task formulation, methods, results table, hop-count analysis
- Slides with the headline screenshot (compare mode showing the failure → success progression)
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
git clone <repo-url> ragscope && cd ragscope

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

# 2. Build the vector index
uv run python scripts/build_index.py

# 3. Sanity check: confirm the LLM doesn't memorize MuSiQue answers
uv run python scripts/memorization_check.py

# 4. Run full evaluation across all methods
uv run python scripts/evaluate.py

# 5. Launch the application
uv run streamlit run src/ragscope/app/streamlit_app.py
```

### Switching methods at runtime

The application's sidebar exposes a method dropdown — selecting a different value reuses the same index and generator and only swaps the retriever. No re-indexing, no model reload.

```python
# In code, methods are interchangeable behind a common protocol:
from ragscope.retrievers import NaiveRetriever, ReRanker, MultiQuery, Decomposition
from ragscope.pipeline import RAGPipeline

pipeline = RAGPipeline(retriever=ReRanker(...), generator=...)
answer, trace = pipeline.run("Who is the spouse of the performer of Imagine?")
```

---

## License & Acknowledgements

- Dataset: MuSiQue (Trivedi et al., 2022) — [github.com/StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)
- Embeddings: BGE (Beijing Academy of AI)
- Re-ranker: MS MARCO MiniLM cross-encoder