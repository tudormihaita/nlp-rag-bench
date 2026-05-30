from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths (relative to repo root; run scripts from there)
    musique_file: Path = Path("data/musique/musique_ans_v1.0_dev.jsonl")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")
    chroma_db_dir: Path = Path("chroma_db")

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator_model: str = "qwen2.5:3b-instruct"

    # Retrieval
    top_k: int = 5
    rerank_candidate_k: int = 30

    # Stratified sampling
    sampling_seed: int = 42
    sample_2hop: int = 200
    sample_3hop: int = 200
    sample_4hop: int = 100

    # Embedding / reranking devices
    embedding_batch_size: int = 128
    embedding_device: str = "mps"
    reranker_device: str = "mps"


settings = Settings()
