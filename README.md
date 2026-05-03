# RAGScope — RAG Methods Benchmark

Coursework project for NLP. A practical benchmark and interactive demo comparing retrieval-augmented generation (RAG) strategies on software documentation Q&A. The central thesis: demonstrating how different retrieval methods affect answer quality, particularly on content that falls outside an LLM's training distribution.

---

## Goal

Build a demo app where a user can:
- Ask a question about a software library's documentation
- Choose a retrieval method (or run all at once for side-by-side comparison)
- See the generated answer, the retrieved chunks, and evaluation scores

The headline demo: one hard question about a post-cutoff or niche feature — the no-RAG baseline hallucinates, naive RAG partially succeeds, the best method answers correctly.

---

## RAG Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **No-RAG baseline** | Bare LLM call, no retrieval — exposes the hallucination/cutoff problem |
| 2 | **Naive RAG** | Chunk → embed → top-k cosine similarity |
| 3 | **Hybrid retrieval** | BM25 + dense embeddings fused via Reciprocal Rank Fusion (RRF) |
| 4 | **Adaptive method** | Cross-encoder re-ranking on top of hybrid retrieval *(or swap: HyDE, multi-query rewriting)* |

---

## Technology Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| **Orchestration** | LlamaIndex `llama-index-core` | Clean retriever-swapping API; can drop down to raw ChromaDB client if abstraction fights back |
| **Vector store** | ChromaDB | In-process, no server needed |
| **Embeddings** | `bge-small-en-v1.5` via `sentence-transformers` | Higher MTEB scores than MiniLM; swap to `bge-micro-v2` if CPU speed is an issue |
| **BM25** | `rank-bm25` | Maintained as a separate in-memory index over raw text |
| **Re-ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Runs on CPU; swap model if latency is a concern |
| **LLM** | Ollama (`llama3.2:3b`, `qwen2.5:3b`, `phi3:mini`) | Local inference, no API key needed; swap model via config |
| **UI** | Streamlit | |
| **Evaluation** | RAGAS | LLM-as-judge metrics; note: RAGAS itself makes LLM calls |
| **Package manager** | `uv` | |

---

## Corpus

**Domain:** Software documentation Q&A.

**Recommended corpus:** Pydantic v2 documentation — the v1→v2 breaking changes are exactly the content an LLM will confidently hallucinate wrong, making for strong adversarial test cases.

**Alternatives:** FastAPI recent versions, LiteLLM docs (post-cutoff), any well-structured MkDocs/Sphinx site.

Target size: 100–500 pages, chunked to ~600 tokens with ~10% overlap.

---

## Project Structure

```
nlp-rag-bench/
├── pyproject.toml
├── .env.example              # API keys template
├── data/
│   ├── raw/                  # scraped HTML / Markdown pages
│   └── processed/            # chunked documents as JSON
├── chroma_db/                # persisted vector store (gitignored)
├── src/
│   └── ragbench/
│       ├── __init__.py
│       ├── config.py         # pydantic-settings: paths, model names, top-k, etc.
│       ├── ingestion/
│       │   ├── scraper.py    # fetch & clean documentation pages
│       │   └── chunker.py    # split into overlapping token-bounded chunks
│       ├── retrievers/
│       │   ├── base.py       # Retriever protocol / abstract base class
│       │   ├── naive.py      # dense top-k cosine similarity
│       │   ├── hybrid.py     # BM25 + dense + RRF fusion
│       │   └── reranking.py  # cross-encoder re-ranking on top of hybrid
│       ├── pipeline.py       # wires retriever + LLM prompt + response parsing
│       ├── evaluation/
│       │   ├── gold_set.json # hand-built Q&A pairs (20–30 questions)
│       │   └── evaluator.py  # RAGAS + hit-rate@k + MRR
│       └── app/
│           └── streamlit_app.py
├── notebooks/
│   └── exploration.ipynb
└── scripts/
    ├── ingest.py             # CLI: scrape → chunk → embed → persist
    └── evaluate.py           # CLI: run all methods over gold set, print table
```

---

## Gold Evaluation Set

20–30 hand-written Q&A pairs split across three tiers:

| Tier | Description | Purpose |
|------|-------------|---------|
| **Lookup** | Answer is verbatim in one chunk | Tests retrieval precision |
| **Synthesis** | Answer requires combining 2–3 chunks | Tests retrieval recall + generation |
| **Adversarial** | Post-cutoff or niche feature, LLM will hallucinate | Headline demo cases |

Each entry in `gold_set.json` includes the question, expected answer, source URL(s), and tier label so metrics can be sliced by difficulty.

---

## Evaluation Metrics

**Retrieval**
- Hit-rate@k — is the relevant chunk in the top-k results?
- MRR — mean reciprocal rank of the first relevant result

**Generation (via RAGAS)**
- Faithfulness — does the answer assert only facts supported by retrieved chunks?
- Answer relevance — is the answer responsive to the question?

**Manual**
- Correctness score (0/1/2) on adversarial tier questions

Results reported as a single comparison table across all four methods.

---

## Implementation Plan

### Phase 1 — Ingestion & Baseline Pipeline
- Scrape and clean corpus, save to `data/raw/`
- Chunk documents with overlap, save to `data/processed/`
- Embed and persist to ChromaDB
- No-RAG and naive RAG working end-to-end via CLI (`scripts/ingest.py`)

### Phase 2 — UI (Streamlit or Gradio, TBD)
- Sidebar: corpus selector, method selector, top-k slider, model selector
- Main panel: question input → answer display → expandable retrieved chunks
- Wire existing pipeline into the UI as a selectable method

### Phase 3 — Retrieval Methods
- Hybrid: build BM25 index over raw chunk text, implement RRF fusion with dense rankings
- Adaptive: cross-encoder re-ranking on the hybrid candidate set *(swap: HyDE or multi-query rewriting)*
- Each method integrated into the UI incrementally

### Phase 4 — Evaluation
- Write gold Q&A set (`evaluation/gold_set.json`)
- Run all methods over the gold set, compute retrieval and RAGAS metrics
- Produce a comparison table sliced by question tier

### Phase 5 — UI Polish & Demo
- Side-by-side comparison tab (same query, all methods)
- Retrieved chunks panel with source metadata, scores, and retrieval trace
- Final comparison table, headline demo screenshot, report / slides

---

## Setup

```bash
# Install dependencies
uv sync

# Install with dev extras
uv sync --extra dev

# Copy and fill in API keys
cp .env.example .env

# Ingest corpus (scrape → chunk → embed)
uv run python scripts/ingest.py

# Launch UI
uv run streamlit run src/ragbench/app/streamlit_app.py
```
