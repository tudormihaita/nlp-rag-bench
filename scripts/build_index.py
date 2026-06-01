#!/usr/bin/env python3
# Author: Alexandru Profir
"""Embed pooled corpus and persist to ChromaDB."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from ragbench.config import settings
from ragbench.indexing.builder import build_index
from ragbench.indexing.embedder import Embedder


def main(args: argparse.Namespace) -> None:
    corpus_path = settings.processed_dir / "pooled_corpus.json"
    if not corpus_path.exists():
        logger.error(f"Corpus not found at {corpus_path}. Run prepare_data.py first.")
        sys.exit(1)

    passages = json.loads(corpus_path.read_text())
    logger.info(f"Loaded {len(passages)} passages from {corpus_path}")

    model = args.model or settings.embedding_model
    device = args.device or settings.embedding_device

    logger.info(f"Loading embedder: {model} on {device}")
    embedder = Embedder(model_name=model, device=device)

    collection = build_index(
        passages=passages,
        embedder=embedder,
        base_path=str(settings.chroma_db_dir),
        batch_size=settings.embedding_batch_size,
        force=args.force,
    )
    logger.success(
        f"Index ready: {collection.count()} passages at "
        f"{settings.chroma_db_dir}/{embedder.slug}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the index even if it already exists",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Embedding model name (overrides RAGBENCH_EMBEDDING_MODEL / config default)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Compute device: cpu | cuda | mps (overrides RAGBENCH_EMBEDDING_DEVICE)",
    )
    main(parser.parse_args())
