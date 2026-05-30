"""
Retrieval and generation metrics for RAG evaluation.

Generation metrics use SQuAD-style normalization (lowercase, strip articles/punctuation)
and are scored against all references (answer + answer_aliases), keeping the best match.

Retrieval metrics compare retrieved doc_ids (ordered) against the gold doc_ids from gold_index.
"""

import re
import string
from collections import Counter


# Utils
def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# Generation metrics
def exact_match(prediction: str, references: list[str]) -> float:
    norm_pred = _normalize(prediction)
    return float(any(_normalize(ref) == norm_pred for ref in references))


def token_f1(prediction: str, references: list[str]) -> float:
    def _f1_single(pred: str, ref: str) -> float:
        pred_tokens = _normalize(pred).split()
        ref_tokens = _normalize(ref).split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((_f1_single(prediction, ref) for ref in references), default=0.0)


# Retrieval metrics
def hit_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Did any gold passage appear in the retrieved set?"""
    gold_set = set(gold_ids)
    return float(any(doc_id in gold_set for doc_id in retrieved_ids))


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of gold passages that appear in the retrieved set."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    return sum(1 for doc_id in retrieved_ids if doc_id in gold_set) / len(gold_ids)


def all_recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Were ALL gold passages retrieved? (strict multi-hop metric)"""
    if not gold_ids:
        return 1.0
    return float(set(gold_ids).issubset(set(retrieved_ids)))


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Mean reciprocal rank of the first gold passage."""
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0
