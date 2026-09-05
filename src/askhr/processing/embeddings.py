# ============================================
# RAG-Research-IQ — Embedding Factory
# Purpose: Create the configured HuggingFace
#          embedding function
# ============================================

from langchain_huggingface import HuggingFaceEmbeddings

from askhr.core.config import settings


def build_embedding_function() -> HuggingFaceEmbeddings:
    """
    Create and return the application's HuggingFace embedding function.

    The embedding model is configured centrally through the application
    settings, keeping model selection separate from the components that
    consume the embeddings.

    Returns:
        HuggingFaceEmbeddings:
            A configured HuggingFace embedding implementation ready to
            generate vector embeddings for documents and queries.

    Configuration:
        - Model name is obtained from ``settings.embedding_model``.
        - ``trust_remote_code`` is enabled to support models that provide
          custom model/tokenizer implementations.

    Notes:
        This function acts as a small factory, allowing the embedding
        implementation or configuration to be changed in one place
        without modifying the vector-store or retrieval components.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "trust_remote_code": True,
            "device": "cpu", # Fast if GPU available; on CPU-only, batching alone still helps somewhat.
        },
        encode_kwargs={
            "batch_size": 8,
            "normalize_embeddings": True,
        },
    )