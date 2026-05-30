import json
from pathlib import Path
from typing import Any


def extract_hop_count(question_id: str) -> int:
    """'3hop1__460_123' → 3.  Always the first character of the ID."""
    return int(question_id[0])


def extract_hop_prefix(question_id: str) -> str:
    """'3hop1__460_123' → '3hop1'.  Preserved as metadata for sub-variant analysis."""
    return question_id.split("__")[0]


def load_musique(path: Path) -> list[dict[str, Any]]:
    """Parse MuSiQue JSONL and enrich each record with hop_count and hop_prefix."""
    questions: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            q["hop_count"] = extract_hop_count(q["id"])
            q["hop_prefix"] = extract_hop_prefix(q["id"])
            questions.append(q)
    return questions