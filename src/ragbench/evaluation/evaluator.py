import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from ragbench.pipeline import RAGPipeline

from .metrics import (
    all_recall_at_k,
    exact_match,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    token_f1,
)

_FIELDNAMES = [
    "question_id",
    "method",
    "hop_count",
    "hop_prefix",
    "question",
    "answer",
    "gold_answer",
    "em",
    "f1",
    "hit_at_k",
    "recall_at_k",
    "all_recall_at_k",
    "mrr",
    "precision_at_k",
    "ndcg_at_k",
    # JSON-encoded list of retrieved passage texts; enables RAGAS scoring as a post-processing pass
    "contexts",
]


def _method_slug(method_name: str) -> str:
    return method_name.lower().replace(" ", "_").replace("-", "_")


def evaluate_method(
    pipeline: RAGPipeline,
    method_name: str,
    questions: list[dict[str, Any]],
    gold_index: dict[str, list[str]],
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """
    Evaluate one retrieval method over all questions with checkpoint/resume.

    Results are appended to `checkpoint_path` as they complete, so interrupted
    runs can be resumed by calling this function again with the same path.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-scored question IDs from a previous (partial) run
    done_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        with open(checkpoint_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done_ids.add(row["question_id"])
        logger.info(
            f"[{method_name}] Resuming from checkpoint; {len(done_ids)}/{len(questions)} already scored"
        )

    remaining = [q for q in questions if q["id"] not in done_ids]
    if not remaining:
        logger.info(f"[{method_name}] Already complete, skipping")
        return rows

    write_header = not checkpoint_path.exists()
    with open(checkpoint_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for q in tqdm(remaining, desc=method_name, unit="q"):
            try:
                result = pipeline.run(q["question"])
            except Exception as exc:
                logger.warning(f"[{method_name}] Error on {q['id']}: {exc!r} — skipping")
                continue
            retrieved_ids = [p.doc_id for p in result.passages]
            gold_ids = gold_index.get(q["id"], [])
            references = [q["answer"]] + q.get("answer_aliases", [])

            contexts = [p.text for p in result.passages]
            row: dict[str, Any] = {
                "question_id": q["id"],
                "method": method_name,
                "hop_count": q["hop_count"],
                "hop_prefix": q["hop_prefix"],
                "question": q["question"],
                "answer": result.answer,
                "gold_answer": q["answer"],
                "em": exact_match(result.answer, references),
                "f1": token_f1(result.answer, references),
                # Retrieval metrics are 0 for No-RAG (empty retrieved_ids)
                "hit_at_k": hit_at_k(retrieved_ids, gold_ids),
                "recall_at_k": recall_at_k(retrieved_ids, gold_ids),
                "all_recall_at_k": all_recall_at_k(retrieved_ids, gold_ids),
                "mrr": mrr(retrieved_ids, gold_ids),
                "precision_at_k": precision_at_k(retrieved_ids, gold_ids),
                "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_ids),
                "contexts": json.dumps(contexts),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)

    logger.success(f"[{method_name}] evaluation complete; {len(rows)} questions scored")
    return rows
