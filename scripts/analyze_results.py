#!/usr/bin/env python3
"""
Aggregate results.csv into per-method and per-method-×-hop summary tables.

Usage:
    uv run python scripts/analyze_results.py
    uv run python scripts/analyze_results.py --input results/results.csv
    uv run python scripts/analyze_results.py --latex
    uv run python scripts/analyze_results.py --latex --output report/tables.tex
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

console = Console()

METRIC_COLS = ["em", "f1", "hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"]
METRIC_HEADERS = ["EM", "F1", "Hit@k", "Rec@k", "AllRec@k", "MRR", "P@k", "NDCG@k"]
RETRIEVAL_ONLY_METRICS = {"hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"}
NO_RAG_METHOD = "No-RAG"
HOPS = [2, 3, 4]


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group(rows: list[dict]) -> dict:
    """Return {method: {hop: [row, ...]}} and {method: [row, ...]}."""
    by_method_hop: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    by_method: dict[str, list] = defaultdict(list)
    for row in rows:
        m = row["method"]
        h = int(row["hop_count"])
        by_method_hop[m][h].append(row)
        by_method[m].append(row)
    return by_method, by_method_hop


def mean_metrics(rows: list[dict], method: str) -> dict[str, float | None]:
    if not rows:
        return {c: None for c in METRIC_COLS}
    result = {}
    for col in METRIC_COLS:
        if col in RETRIEVAL_ONLY_METRICS and method == NO_RAG_METHOD:
            result[col] = None
        else:
            result[col] = sum(float(r[col]) for r in rows) / len(rows)
    return result


def fmt(v: float | None, col: str, method: str) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


# ── Rich console tables ────────────────────────────────────────────────────────

def print_overall(by_method: dict, methods: list[str]) -> None:
    t = Table(title="Overall averages (all hop counts)", show_header=True, header_style="bold cyan")
    t.add_column("Method", style="bold", min_width=22)
    for h in METRIC_HEADERS:
        t.add_column(h, justify="right", min_width=8)

    for m in methods:
        stats = mean_metrics(by_method.get(m, []), m)
        t.add_row(m, *[fmt(stats[c], c, m) for c in METRIC_COLS])
    console.print(t)


def print_by_hop(by_method_hop: dict, methods: list[str]) -> None:
    for metric_group, cols, headers in [
        ("Generation metrics", ["em", "f1"], ["EM", "F1"]),
        ("Retrieval metrics", ["hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"],
         ["Hit@k", "Rec@k", "AllRec@k", "MRR", "P@k", "NDCG@k"]),
    ]:
        t = Table(title=f"{metric_group} — per method × hop count", show_header=True, header_style="bold magenta")
        t.add_column("Method", style="bold", min_width=22)
        for hop in HOPS:
            for h in headers:
                t.add_column(f"{h} ({hop}h)", justify="right", min_width=8)

        for m in methods:
            row_cells = [m]
            for hop in HOPS:
                hop_rows = by_method_hop.get(m, {}).get(hop, [])
                stats = mean_metrics(hop_rows, m)
                for c in cols:
                    row_cells.append(fmt(stats[c], c, m))
            t.add_row(*row_cells)
        console.print(t)


def print_sample_counts(by_method_hop: dict, methods: list[str]) -> None:
    t = Table(title="Sample counts", show_header=True, header_style="bold dim")
    t.add_column("Method", style="bold", min_width=22)
    for hop in HOPS:
        t.add_column(f"{hop}-hop", justify="right")
    t.add_column("Total", justify="right", style="bold")

    for m in methods:
        counts = [len(by_method_hop.get(m, {}).get(hop, [])) for hop in HOPS]
        t.add_row(m, *[str(c) for c in counts], str(sum(counts)))
    console.print(t)


# ── LaTeX export ───────────────────────────────────────────────────────────────

def _tex_cell(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def latex_overall(by_method: dict, methods: list[str]) -> str:
    cols = len(METRIC_COLS)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Overall average metrics across all hop counts}",
        r"\label{tab:overall}",
        r"\small",
        r"\begin{tabular}{l" + "r" * cols + "}",
        r"\toprule",
        "Method & " + " & ".join(METRIC_HEADERS) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        stats = mean_metrics(by_method.get(m, []), m)
        cells = [_tex_cell(stats[c]) for c in METRIC_COLS]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_by_hop(by_method_hop: dict, methods: list[str]) -> str:
    sections = []
    for label, cols, headers, tag in [
        ("Generation metrics per method and hop count", ["em", "f1"], ["EM", "F1"], "gen"),
        ("Retrieval metrics per method and hop count",
         ["hit_at_k", "recall_at_k", "all_recall_at_k", "mrr", "precision_at_k", "ndcg_at_k"],
         ["Hit@k", "Rec@k", "AllRec@k", "MRR", "P@k", "NDCG@k"], "ret"),
    ]:
        n_data_cols = len(HOPS) * len(cols)
        col_spec = "l" + "r" * n_data_cols

        # Build header: one \multicolumn per hop
        hop_headers = " & ".join(
            rf"\multicolumn{{{len(cols)}}}{{c}}{{{hop}-hop}}" for hop in HOPS
        )
        sub_headers = " & ".join(h for _ in HOPS for h in headers)

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{{label}}}",
            rf"\label{{tab:{tag}_by_hop}}",
            r"\small",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            f"Method & {hop_headers} " + r"\\",
            f" & {sub_headers} " + r"\\",
            r"\midrule",
        ]
        for m in methods:
            cells = []
            for hop in HOPS:
                hop_rows = by_method_hop.get(m, {}).get(hop, [])
                stats = mean_metrics(hop_rows, m)
                for c in cols:
                    cells.append(_tex_cell(stats[c]))
            lines.append(f"{m} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def main(args: argparse.Namespace) -> None:
    path = Path(args.input)
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)

    rows = load_rows(path)
    if not rows:
        console.print("[red]results.csv is empty.[/red]")
        sys.exit(1)

    by_method, by_method_hop = group(rows)
    # Preserve method order from the CSV (first-seen order)
    seen: dict[str, None] = {}
    for r in rows:
        seen[r["method"]] = None
    methods = list(seen)

    console.print(f"\n[bold]Loaded {len(rows)} rows · {len(methods)} methods · {path}[/bold]\n")

    print_sample_counts(by_method_hop, methods)
    console.print()
    print_overall(by_method, methods)
    console.print()
    print_by_hop(by_method_hop, methods)

    if args.latex:
        tex = latex_overall(by_method, methods) + "\n\n" + latex_by_hop(by_method_hop, methods)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(tex)
            console.print(f"\n[green]LaTeX tables written to {out}[/green]")
        else:
            console.print("\n[bold]LaTeX output:[/bold]")
            console.print(tex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/results.csv", metavar="PATH",
        help="Path to results CSV (default: results/results.csv)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also emit LaTeX table source",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="Write LaTeX to this file instead of stdout (e.g. report/tables.tex)",
    )
    main(parser.parse_args())
