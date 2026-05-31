#!/usr/bin/env python3
"""
Sanity check: run all available methods on N random questions and display
per-question outputs and metrics. Writes nothing to disk — use before full eval.

Usage:
    uv run python scripts/sanity_check.py
    uv run python scripts/sanity_check.py --n 10 --hop 3
    uv run python scripts/sanity_check.py --methods "Classic RAG" "Re-ranking"
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from rich.console import Console
from rich.table import Table

from ragbench.config import settings
from ragbench.evaluation.metrics import (
    all_recall_at_k,
    exact_match,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)
from ragbench.factory import PipelineName, build_pipelines
from ragbench.generation.llm import Generator
from ragbench.indexing.builder import index_exists
from ragbench.indexing.embedder import Embedder
from ragbench.pipeline import RAGPipeline

console = Console(highlight=False)


def _load_pipelines(embedder: Embedder, gen: Generator) -> dict[str, RAGPipeline]:
    if not index_exists(settings.embedding_model, str(settings.chroma_db_dir)):
        console.print("[yellow]Vector index not found — only No-RAG available. Build index first.[/yellow]")
        from ragbench.pipeline import RAGPipeline
        return {PipelineName.NO_RAG: RAGPipeline(generator=gen, retriever=None, top_k=settings.top_k)}
    return build_pipelines(embedder, gen, settings)


def render_question(
    q: dict,
    gold_ids: list[str],
    pipelines: dict[str, RAGPipeline],
    methods: list[str],
) -> None:
    from ragbench.pipeline import PipelineResult

    references = [q["answer"]] + q.get("answer_aliases", [])
    gold_set = set(gold_ids)

    console.rule(f"[bold white]{q['id']}  ({q['hop_count']}-hop)[/bold white]")
    console.print(f"[bold cyan]Question:[/bold cyan] {q['question']}")
    console.print(f"[bold green]Gold answer:[/bold green] {q['answer']}")
    if q.get("answer_aliases"):
        console.print(f"[dim]Aliases: {', '.join(q['answer_aliases'])}[/dim]")
    console.print()

    # Run each method once; reuse results for both tables
    runs: list[tuple[str, PipelineResult]] = [
        (name, pipelines[name].run(q["question"]))
        for name in methods
        if name in pipelines
    ]

    # Metrics table
    metrics_table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    metrics_table.add_column("Method", min_width=14)
    metrics_table.add_column("Answer", min_width=30, max_width=50, no_wrap=False)
    metrics_table.add_column("EM", justify="center", min_width=4)
    metrics_table.add_column("F1", justify="right", min_width=5)
    metrics_table.add_column("Hit", justify="right", min_width=5)
    metrics_table.add_column("Rec", justify="right", min_width=5)
    metrics_table.add_column("AllRec", justify="right", min_width=7)
    metrics_table.add_column("NDCG", justify="right", min_width=5)
    metrics_table.add_column("Prec", justify="right", min_width=5)

    for method_name, result in runs:
        retrieved_ids = [p.doc_id for p in result.passages]
        em = exact_match(result.answer, references)
        f1 = token_f1(result.answer, references)

        if retrieved_ids:
            hit = f"{hit_at_k(retrieved_ids, gold_ids):.2f}"
            rec = f"{recall_at_k(retrieved_ids, gold_ids):.2f}"
            allrec = f"{all_recall_at_k(retrieved_ids, gold_ids):.2f}"
            ndcg = f"{ndcg_at_k(retrieved_ids, gold_ids):.2f}"
            prec = f"{precision_at_k(retrieved_ids, gold_ids):.2f}"
        else:
            hit = rec = allrec = ndcg = prec = "—"

        em_cell = "[bold green]✓[/bold green]" if em == 1.0 else "[red]✗[/red]"
        metrics_table.add_row(
            method_name, result.answer.strip()[:120], em_cell, f"{f1:.2f}",
            hit, rec, allrec, ndcg, prec,
        )

    console.print(metrics_table)

    # Decomposition traces (only for methods that produced sub-questions)
    _DECOMP_METHODS = {PipelineName.DECOMPOSITION, PipelineName.ITERATIVE_DECOMPOSITION}
    decomp_runs = [(name, result) for name, result in runs if name in _DECOMP_METHODS and result.trace.sub_queries]
    if decomp_runs:
        console.print()
        for method_name, result in decomp_runs:
            t = result.trace
            console.print(f"[bold dim]{method_name} trace[/bold dim]")
            for i, sq in enumerate(t.sub_queries, 1):
                enriched = t.enriched_queries[i - 1] if i <= len(t.enriched_queries) else None
                ans = t.intermediate_answers[i - 1] if i <= len(t.intermediate_answers) else None
                sq_label = f"  [dim]{i}.[/dim] {sq}"
                if enriched and enriched != sq:
                    sq_label += f"\n     [cyan]→ {enriched}[/cyan]"
                if ans:
                    sq_label += f"\n     [green]✦ {ans}[/green]"
                console.print(sq_label)

    # Retrieved passages — one column per method, one row per rank (★ = gold, dim = distractor)
    passage_table = Table(show_header=True, header_style="bold dim", box=None, pad_edge=False)
    passage_table.add_column("#", justify="right", min_width=2, style="dim")
    for method_name, result in runs:
        has_passages = bool(result.passages)
        passage_table.add_column(
            method_name if has_passages else f"{method_name} [dim](no retrieval)[/dim]",
            min_width=18,
            max_width=32,
            no_wrap=True,
        )

    max_passages = max((len(result.passages) for _, result in runs), default=0)
    for i in range(max_passages):
        row_cells: list[str] = [str(i + 1)]
        for _, result in runs:
            if i >= len(result.passages):
                row_cells.append("[dim]—[/dim]")
            else:
                p = result.passages[i]
                if p.doc_id in gold_set:
                    row_cells.append(f"[bold yellow]★[/bold yellow] {p.title}")
                else:
                    row_cells.append(f"[dim]{p.title}[/dim]")
        passage_table.add_row(*row_cells)

    console.print()
    console.print("[bold dim]Retrieved passages (★ = gold)[/bold dim]")
    console.print(passage_table)
    console.print()


def main(args: argparse.Namespace) -> None:
    questions_path = settings.processed_dir / "sampled_questions.json"
    gold_path = settings.processed_dir / "gold_index.json"

    for p in (questions_path, gold_path):
        if not p.exists():
            logger.error(f"Missing {p}. Preprocess data first.")
            sys.exit(1)

    questions: list[dict] = json.loads(questions_path.read_text())
    gold_index: dict[str, list[str]] = json.loads(gold_path.read_text())

    if args.hop:
        questions = [q for q in questions if q["hop_count"] == args.hop]
        if not questions:
            logger.error(f"No {args.hop}-hop questions found after filtering.")
            sys.exit(1)

    rng = random.Random(args.seed)
    sample = rng.sample(questions, min(args.n, len(questions)))
    logger.info(f"Sampled {len(sample)} questions (seed={args.seed})")

    logger.info("Loading models...")
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    gen = Generator(
        settings.generator_model,
        host=settings.api_url,
        auth_bearer=settings.api_auth_bearer,
        api_src=settings.api_src,
    )
    pipelines = _load_pipelines(embedder, gen)

    methods = args.methods or list(pipelines.keys())
    unknown = [m for m in methods if m not in pipelines]
    if unknown:
        logger.warning(f"Unknown methods (will skip): {unknown}")
        methods = [m for m in methods if m in pipelines]

    console.print(f"\n[bold]Smoke test — {len(sample)} questions, methods: {', '.join(methods)}[/bold]\n")

    for q in sample:
        gold_ids = gold_index.get(q["id"], [])
        render_question(q, gold_ids, pipelines, methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="Number of questions (default: 5)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for question selection (default: 1)")
    parser.add_argument("--hop", type=int, choices=[2, 3, 4], default=None, help="Filter to a specific hop count")
    parser.add_argument(
        "--methods", nargs="+", default=None, metavar="METHOD",
        help='Methods to run (default: all). E.g. --methods "Classic RAG" "Re-ranking"'
    )
    main(parser.parse_args())
