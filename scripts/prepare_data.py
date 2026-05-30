#!/usr/bin/env python3
"""Load MuSiQue dev split, stratified sample, pool corpus and save to data/processed/."""
import argparse
import json
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from ragbench.config import settings
from ragbench.data.corpus import build_corpus
from ragbench.data.loader import load_musique
from ragbench.data.sampler import sample_stratified


def main(args: argparse.Namespace) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading MuSiQue from {settings.musique_file}")
    questions = load_musique(settings.musique_file)
    logger.info(f"Loaded {len(questions)} questions total")

    n_per_hop = {
        2: settings.sample_2hop,
        3: settings.sample_3hop,
        4: settings.sample_4hop,
    }
    sampled = sample_stratified(questions, n_per_hop, seed=settings.sampling_seed)
    hop_counts = {h: sum(1 for q in sampled if q["hop_count"] == h) for h in n_per_hop}
    logger.info(f"Sampled {len(sampled)} questions — by hop: {hop_counts}")

    passages, gold_index = build_corpus(sampled)
    logger.info(f"Pooled corpus: {len(passages)} unique passages")

    sampled_path = settings.processed_dir / "sampled_questions.json"
    corpus_path = settings.processed_dir / "pooled_corpus.json"
    gold_path = settings.processed_dir / "gold_index.json"

    sampled_path.write_text(json.dumps(sampled, indent=2))
    corpus_path.write_text(json.dumps(passages, indent=2))
    gold_path.write_text(json.dumps(gold_index, indent=2))

    logger.success(f"Saved sampled_questions ({len(sampled)}), pooled_corpus ({len(passages)}), gold_index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
