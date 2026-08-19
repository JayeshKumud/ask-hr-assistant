import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Embeddings ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- LLM ---
    llm_model: str = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    chunk_separators: tuple = ("\n\n", "\n", ".", " ")

    # --- Vector store ---
    vector_store_dir: Path = Path(os.getenv("VECTOR_STORE_DIR", "resources/vectorstore"))
    collection_name: str = os.getenv("COLLECTION_NAME", "real_estate")

    # --- Retrieval / QA ---
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_tokens_limit: int = int(os.getenv("MAX_TOKENS_LIMIT", "8000"))


settings = Settings()
