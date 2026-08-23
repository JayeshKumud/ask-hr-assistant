from typing import List, Tuple, Iterator, Optional

from langchain_groq import ChatGroq

from core.config import settings
from processing.vector_store import VectorStoreManager
from ingestion.document_loader import load_documents
from ingestion.text_splitter import split_documents
from search.citations import Citation
from search.qa_chain import build_qa_chain, generate_answer as _generate_answer


class RAGPipeline:
    """
    Orchestrates ingestion (load policy documents -> split -> store) and
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
                # qwen3 models reason internally by default, and that
                # reasoning consumes tokens from the SAME max_tokens
                # budget as the final answer. reasoning_format="hidden"
                # (an earlier attempt) only hides the reasoning text from
                # the output — it doesn't stop the model from spending
                # tokens on it, which is what caused empty answers when
                # max_tokens was too small. reasoning_effort="none" fixes
                # this at the root: it disables internal reasoning
                # entirely, so the full token budget goes to the actual
                # answer. Appropriate here since this is a narrow
                # retrieval-grounded QA task, not a multistep reasoning
                # problem — the retrieved chunks are already doing the
                # "thinking".
                reasoning_effort="none",
            )
        return self._llm

    def ingest_documents(self) -> Iterator[str]:
        """
        Loads, splits, and stores the company policy documents from
        settings.policies_dir. Yields progress messages (same generator
        pattern as the old process_urls()).
        """
        yield "Resetting vector store...✅"
        self._vsm.reset()

        yield "Loading policy documents...✅"
        data = load_documents()

        yield "Splitting text into chunks...✅"
        docs = split_documents(data)

        yield "Adding chunks to vector database...✅"
        self._vsm.add_documents(docs)

        yield "Done adding docs to vector database...✅"

    def generate_answer(self, query: str) -> Tuple[str, List[Citation]]:
        if self._vsm.store is None:
            raise RuntimeError("Vector database is not initialized")

        chain = build_qa_chain(self.llm, self._vsm)
        return _generate_answer(chain, query)


if __name__ == "__main__":
    # Manual smoke test: ingests the real policy documents, asks one
    # question, prints the answer + citations. Needs GROQ_API_KEY set in
    # .env. Run with: python -m core.pipeline (from src/, with the venv
    # active) — confirms ingestion + retrieval + generation + citations
    # all work together before you commit.
    pipeline = RAGPipeline()

    for status in pipeline.ingest_documents():
        print(status)

    answer, citations = pipeline.generate_answer(
        "How many annual leave days does a full-time NexaCore employee receive?"
    )
    print(f"Answer: {answer}")
    print(f"\n{len(citations)} citation(s):")
    for c in citations:
        print(f"- {c.display_label()}")
        print(f"    {c.snippet[:150]}...")