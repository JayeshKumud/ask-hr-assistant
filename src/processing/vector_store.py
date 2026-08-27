# ============================================
# RAG-Research-IQ — Vector Store Manager
# Purpose: Manage Chroma lifecycle, persistence,
#          document storage, and retrieval
# ============================================

from uuid import uuid4
from typing import Iterable, Optional, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import settings
from processing.embeddings import build_embedding_function


class VectorStoreManager:
    """
    Manage the lifecycle and operations of the Chroma vector store.

    This class provides a single abstraction for:
        - Lazy initialization of the Chroma vector store.
        - Persistent vector storage.
        - Collection reset operations.
        - Adding documents with unique IDs.
        - Creating a LangChain retriever.

    The embedding function can be injected through the constructor,
    which makes the class easier to test and allows different embedding
    implementations to be used without changing the vector-store logic.
    """

    def __init__(self, embedding_function: Optional[Any] = None) -> None:
        """
        Initialize the vector store manager.

        Args:
            embedding_function:
                Optional embedding implementation. If not provided,
                the application's configured embedding function is created
                using ``build_embedding_function()``.

        Notes:
            The Chroma store itself is initialized lazily when the
            ``store`` property is first accessed.
        """
        self._embedding_function = (
            embedding_function
            if embedding_function is not None
            else build_embedding_function()
        )
        self._store: Optional[Chroma] = None

    @property
    def store(self) -> Chroma | None:
        """
        Return the Chroma vector store, creating it if necessary.

        The Chroma store is initialized only on first access. This avoids
        unnecessary initialization when the manager is created but the
        vector store is not yet required.

        Returns:
            Chroma:
                The configured Chroma vector store instance.

        Configuration:
            - Collection name is obtained from ``settings.collection_name``.
            - Embeddings are provided by the configured embedding function.
            - Persistence directory is obtained from
              ``settings.vector_store_dir``.
        """
        if self._store is None:
            self._store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self._embedding_function,
                persist_directory=str(settings.vector_store_dir),
            )

        return self._store

    def reset(self) -> None:
        """
        Reset the configured Chroma collection.

        This removes the existing collection data and is typically used
        during development, testing, or before a complete re-ingestion
        of source documents.
        """
        self.store.reset_collection()

    def add_documents(self, docs: Iterable[Document]) -> list[str]:
        """
        Add documents to the Chroma vector store.

        Each document receives a newly generated UUID-based ID to avoid
        ID collisions across ingestion runs.

        Args:
            docs:
                An iterable of LangChain ``Document`` objects. Generators
                and other one-time iterables are supported.

        Returns:
            list[str]:
                The IDs assigned to the added documents.
        """
        documents = list(docs)
        ids = [str(uuid4()) for _ in documents]

        self.store.add_documents(documents, ids=ids)

        return ids

    def as_retriever(self, k: Optional[int] = None) -> Any:
        """
        Create a LangChain retriever backed by the Chroma store.

        Args:
            k:
                Optional override for how many documents to return.
                Defaults to ``settings.top_k``. The re-ranking pipeline
                (search/reranker.py) passes a wider value here — it needs
                a bigger candidate pool to narrow down from, since
                re-ranking a pool the same size as the final result
                accomplishes nothing.

        Returns:
            Any:
                A LangChain retriever configured for semantic search.
        """
        return self.store.as_retriever(
            search_kwargs={"k": k or settings.top_k}
        )

    def get_all_documents(self) -> list[Document]:
        """
        Reconstruct every chunk currently stored in the collection as a
        list of LangChain Documents.

        Unlike Chroma, BM25 isn't a persisted index — BM25Retriever needs
        the full text corpus in memory to build its keyword index. Rather
        than maintaining a separate copy of the chunks that could drift
        out of sync with what's actually in Chroma (e.g. after a reset +
        re-ingest), this reads the current corpus directly back out of
        Chroma on demand, so BM25 always reflects whatever is actually
        stored right now.

        Returns:
            list[Document]:
                Every chunk in the collection, with its original
                page_content and metadata restored.
        """
        raw = self.store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(raw["documents"], raw["metadatas"])
        ]