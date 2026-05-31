from pathlib import Path

import chromadb
from loguru import logger

from .embedder import Embedder

COLLECTION_NAME = "musique"
_CHROMA_BATCH = 5_000  # ChromaDB internal limit for batch size retrieved in a single call


def index_exists(model_name: str, base_path: str = "chroma_db") -> bool:
    slug = model_name.split("/")[-1]
    return Path(f"{base_path}/{slug}").exists()


def load_collection(model_name: str, base_path: str = "chroma_db") -> chromadb.Collection:
    slug = model_name.split("/")[-1]
    path = f"{base_path}/{slug}"
    client = chromadb.PersistentClient(path=path)
    return client.get_collection(COLLECTION_NAME)


def build_index(
    passages: list[dict],
    embedder: Embedder,
    base_path: str = "chroma_db",
    batch_size: int = 64,
    force: bool = False,
) -> chromadb.Collection:
    """
    Embed passages and persist them to a ChromaDB collection.

    The collection lives at <base_path>/<embedder.slug>/ so multiple embedders
    can coexist without overwriting each other.

    Args:
        force: if True, delete and rebuild even when the collection already exists.
               if False (default), return the existing collection unchanged.
    """
    path = f"{base_path}/{embedder.slug}"
    client = chromadb.PersistentClient(path=path)

    try:
        existing = client.get_collection(COLLECTION_NAME)
        if not force:
            logger.info(
                f"Collection '{COLLECTION_NAME}' already exists at {path} "
                f"({existing.count()} passages). Skipping. Pass --force to rebuild."
            )
            return existing
        logger.info(f"Force rebuild: deleting existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [p["text"] for p in passages]
    logger.info(f"Encoding {len(texts)} passages with {embedder.model_name}")
    embeddings = embedder.encode_passages(texts, batch_size=batch_size)

    # ChromaDB has no support for list-typed metadata; join source_qids as CSV
    metadatas = [{"title": p["title"], "source_qids": ",".join(p["source_qids"])} for p in passages]

    for start in range(0, len(passages), _CHROMA_BATCH):
        end = min(start + _CHROMA_BATCH, len(passages))
        collection.add(
            ids=[p["doc_id"] for p in passages[start:end]],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        logger.info(f"Indexed {end}/{len(passages)} passages")

    return collection
