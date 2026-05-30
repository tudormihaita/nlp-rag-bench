import hashlib
from typing import Any


def _doc_id(text: str) -> str:
    """Unique ID method for corpus documents to enable deduplication: first 12 hex chars of SHA-1(text)."""
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def build_corpus(
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """
    Pool and deduplicate paragraphs across all sampled questions.

    Deduplication is by content hash, so a paragraph shared across multiple
    questions appears once in the corpus with all source question IDs recorded.

    Returns:
        passages:    list of {doc_id, title, text, source_qids}
        gold_index:  {question_id: [doc_id, ...]} supporting passages per question
    """
    corpus: dict[str, dict[str, Any]] = {}
    gold_index: dict[str, list[str]] = {}

    for q in questions:
        qid = q["id"]
        gold_ids: list[str] = []

        for para in q["paragraphs"]:
            text = para["paragraph_text"]
            doc_id = _doc_id(text)

            if doc_id not in corpus:
                corpus[doc_id] = {
                    "doc_id": doc_id,
                    "title": para["title"],
                    "text": text,
                    "source_qids": [qid],
                }
            elif qid not in corpus[doc_id]["source_qids"]:
                corpus[doc_id]["source_qids"].append(qid)

            if para["is_supporting"]:
                gold_ids.append(doc_id)

        gold_index[qid] = gold_ids

    return list(corpus.values()), gold_index
