#!/usr/bin/env python3
# Author: Alexandru Profir
"""
Run all 4 retrieval methods over the sampled MuSiQue questions and print
the headline comparison table. Results are checkpointed per method so
interrupted runs can be resumed.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --methods "No-RAG" "Classic RAG"
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

from loguru import logger
from rich.console import Console
from rich.table import Table

from ragbench.config import settings
from ragbench.evaluation.evaluator import _method_slug, evaluate_method
from ragbench.factory import PipelineName, build_pipelines
from ragbench.generation.llm import Generator
from ragbench.indexing.embedder import Embedder

console = Console()


def aggregate(all_rows: list[dict]) -> dict[tuple[str, int], dict[str, float]]:
    """Group rows by (method, hop_count) and compute mean metrics."""
    from collections import defaultdict
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in all_rows:
        groups[(row["method"], int(row["hop_count"]))].append(row)

    result = {}
    float_cols = ["em", "f1", "hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"]
    for key, rows in groups.items():
        result[key] = {col: sum(float(r[col]) for r in rows) / len(rows) for col in float_cols}
    return result


def print_table(agg: dict[tuple[str, int], dict[str, float]], methods: list[str]) -> None:
    hops = [2, 3, 4]

    def cell(method: str, hop: int, col: str) -> str:
        v = agg.get((method, hop), {}).get(col)
        return f"{v:.3f}" if v is not None else "—"

    # Generation table
    gen_table = Table(title="Generation Metrics", show_header=True, header_style="bold cyan")
    gen_table.add_column("Method", style="bold")
    for hop in hops:
        gen_table.add_column(f"EM ({hop}h)", justify="right")
        gen_table.add_column(f"F1 ({hop}h)", justify="right")
    for method in methods:
        gen_table.add_row(
            method,
            *[cell(method, hop, col) for hop in hops for col in ["em", "f1"]],
        )
    console.print(gen_table)

    # Retrieval table (No-RAG always shows — since no passages retrieved)
    ret_cols = ["hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"]
    ret_headers = ["Hit", "Rec", "AllRec", "MRR", "Prec", "NDCG"]
    ret_table = Table(title="Retrieval Metrics", show_header=True, header_style="bold magenta")
    ret_table.add_column("Method", style="bold")
    for hop in hops:
        for header in ret_headers:
            ret_table.add_column(f"{header} ({hop}h)", justify="right")
    for method in methods:
        if method == PipelineName.NO_RAG:
            ret_table.add_row(method, *["—"] * (len(hops) * len(ret_cols)))
        else:
            ret_table.add_row(
                method,
                *[cell(method, hop, col) for hop in hops for col in ret_cols],
            )
    console.print(ret_table)


def main(args: argparse.Namespace) -> None:
    # Load processed data
    questions_path = settings.processed_dir / "sampled_questions.json"
    gold_path = settings.processed_dir / "gold_index.json"
    for p in (questions_path, gold_path):
        if not p.exists():
            logger.error(f"Missing {p}. Preprocess data first.")
            sys.exit(1)

    questions = json.loads(questions_path.read_text())
    gold_index = json.loads(gold_path.read_text())
    logger.info(f"Loaded {len(questions)} questions, {len(gold_index)} gold entries")

    # Build pipelines
    logger.info("Loading embedder and generator...")
    embedder = Embedder(settings.embedding_model, settings.embedding_device)
    gen = Generator(
        settings.generator_model,
        host=settings.api_url,
        auth_bearer=settings.api_auth_bearer,
        api_src=settings.api_src,
    )
    pipelines = build_pipelines(embedder, gen, settings)

    methods = args.methods or list(pipelines.keys())
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each method (with checkpoint/resume)
    all_rows: list[dict] = []
    for method in methods:
        if method not in pipelines:
            logger.warning(f"Unknown method '{method}', skipping")
            continue
        checkpoint = settings.results_dir / f"{_method_slug(method)}.csv"
        rows = evaluate_method(
            pipeline=pipelines[method],
            method_name=method,
            questions=questions,
            gold_index=gold_index,
            checkpoint_path=checkpoint,
        )
        all_rows.extend(rows)

    # Merge all method CSVs into one results file
    results_path = settings.results_dir / "results.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.success(f"Saved {len(all_rows)} rows to {results_path}")

    # Print headline table
    agg = aggregate(all_rows)
    print_table(agg, methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help='Methods to evaluate (default: all). E.g. --methods "No-RAG" "Classic RAG"',
    )
    main(parser.parse_args())
