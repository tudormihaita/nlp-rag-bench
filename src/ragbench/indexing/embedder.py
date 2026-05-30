import numpy as np
from sentence_transformers import SentenceTransformer

# Models that require task-specific input prefixes for correct embedding behavior
PREFIXED_MODELS: dict[str, dict[str, str]] = {
    "intfloat/e5-base-v2":            {"query": "query: ",        "passage": "passage: "},
    "intfloat/e5-small-v2":           {"query": "query: ",        "passage": "passage: "},
    "nomic-ai/nomic-embed-text-v1.5": {"query": "search_query: ", "passage": "search_document: "},
}


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.prefixes = PREFIXED_MODELS.get(model_name, {"query": "", "passage": ""})

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [self.prefixes["passage"] + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.prefixes["query"] + text,
            normalize_embeddings=True,
        )

    @property
    def slug(self) -> str:
        """Model identifier used as the ChromaDB subfolder name."""
        return self.model_name.split("/")[-1]
