import random
from typing import Any


def sample_stratified(
    questions: list[dict[str, Any]],
    n_per_hop: dict[int, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Draw n_per_hop[k] questions for each hop count k, without replacement.
    Uses seed so the sample is deterministic and reproducible.
    """
    by_hop: dict[int, list[dict[str, Any]]] = {}
    for q in questions:
        by_hop.setdefault(q["hop_count"], []).append(q)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for hop_count, n in sorted(n_per_hop.items()):
        pool = by_hop.get(hop_count, [])
        if len(pool) < n:
            raise ValueError(
                f"Requested {n} {hop_count}-hop questions but only {len(pool)} available"
            )
        sampled.extend(rng.sample(pool, n))

    return sampled
