import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from core.logging import configure_logging

configure_logging(enabled=False)
load_dotenv()

# Anchor all default relative paths to the project root, not the
# process's current working directory. cwd varies depending on how the
# script is launched — `python -m core.pipeline` from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass(frozen=True)
class Settings:
    # --- Embeddings ---
    os.environ["HF_HUB_OFFLINE"] = "1"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- LLM ---
    llm_groq_qwen_model: str = os.getenv("LLM_MODEL", "NON_EXISTING_qwen/qwen3.6-27b")
    llm_groq_provider: str = os.getenv("LLM_GROQ_PROVIDER", "groq")

    llm_huggingface_google_model: str = os.getenv("LLM_HUGGINGFACE_GOOGLE_MODEL", "google/gemma-2-2b-it:featherless-ai")
    llm_huggingface_google_provider: str = os.getenv("LLM_HUGGINGFACE_GOOGLE_PROVIDER", "huggingface")
    llm_huggingface_mistral_model: str = os.getenv("LLM_HUGGINGFACE_MISTRAL_MODEL", "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai")
    llm_huggingface_mistral_provider: str = os.getenv("LLM_HUGGINGFACE_MISTRAL_PROVIDER", "huggingface")

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    chunk_separators: tuple = ("\n\n", "\n", ".", " ")

    # --- Vector store ---
    # PROJECT_ROOT / <relative path> resolves normally; if VECTOR_STORE_DIR
    # is set to an absolute path via env var instead, pathlib's `/`
    # operator discards PROJECT_ROOT automatically and uses the absolute
    # path as-is — no special-casing needed for that case.
    vector_store_dir: Path = PROJECT_ROOT / os.getenv("VECTOR_STORE_DIR", "resources/vectorstore")
    collection_name: str = os.getenv("COLLECTION_NAME", "company_policies")

    # --- Ingestion ---
    # Directory containing the source policy PDFs that get loaded,
    # chunked, and indexed. Replaces the old URL-list approach.
    policies_dir: Path = PROJECT_ROOT / os.getenv("POLICIES_DIR", "resources/policies")

    # --- Retrieval / QA ---
    top_k: int = int(os.getenv("TOP_K", "3"))
    max_tokens_limit: int = int(os.getenv("MAX_TOKENS_LIMIT", "8000"))

    # --- Hybrid retrieval ---
    # Relative weight given to each retriever when combining BM25
    # (keyword/exact-match) and vector (semantic) search results via
    # EnsembleRetriever's Reciprocal Rank Fusion. Must sum to 1.0.
    # BM25 tends to win on exact terms (form numbers, specific phrases
    # like "25 working days"); vector search tends to win on paraphrased
    # or conceptually-related questions that don't share exact wording
    # with the source text. 0.5/0.5 is a neutral starting point —
    # Phase 6's eval script is what should actually tell you whether to
    # shift this.
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.5"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))

    # --- Re-ranking ---
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # How many candidates hybrid retrieval fetches BEFORE the cross-encoder
    # re-ranks and narrows them down to top_k for the final prompt. This
    # must be wider than top_k, or there's nothing for the re-ranker to
    # actually narrow — re-ranking 5 candidates down to 5 does nothing.
    rerank_candidate_k: int = int(os.getenv("RERANK_CANDIDATE_K", "15"))


settings = Settings()