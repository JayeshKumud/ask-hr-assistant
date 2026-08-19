from typing import List, Tuple, Iterator, Optional

from langchain_groq import ChatGroq

from core.config import settings
from processing.vector_store import VectorStoreManager
from ingestion.url_loader import load_urls
from ingestion.text_splitter import split_documents
from search.qa_chain import build_qa_chain, generate_answer as _generate_answer


class RAGPipeline:
    """
    Orchestrates ingestion (load URLs -> split -> store) and
    retrieval (query -> answer + sources). Replaces module-level
    globals + initialize_components() with instance state, which
    is easier to test and reason about.
    """

    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        self._llm = None
        self._vsm = vector_store_manager or VectorStoreManager()

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        return self._llm

    def process_urls(self, urls: List[str]) -> Iterator[str]:
        """
        Loads, splits, and stores documents from URLs.
        Yields progress messages (same generator pattern as original).
        """
        yield "Resetting vector store...✅"
        self._vsm.reset()

        yield "Loading data...✅"
        data = load_urls(urls)

        yield "Splitting text into chunks...✅"
        docs = split_documents(data)

        yield "Adding chunks to vector database...✅"
        self._vsm.add_documents(docs)

        yield "Done adding docs to vector database...✅"

    def generate_answer(self, query: str) -> Tuple[str, List[str]]:
        if self._vsm.store is None:
            raise RuntimeError("Vector database is not initialized")

        chain = build_qa_chain(self.llm, self._vsm.store)
        return _generate_answer(chain, query)
