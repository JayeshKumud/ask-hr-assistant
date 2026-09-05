from typing import List, Tuple, Iterator, Optional, Any, Sequence

from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableWithFallbacks

from askhr.core.config import settings
from askhr.core.llm_wrappers import build_llm_with_fallback
from askhr.ingestion.document_loader import load_documents
from askhr.ingestion.text_splitter import split_documents
from askhr.processing.vector_store import VectorStoreManager
from askhr.search.citations import Citation
from langchain_classic.chains import RetrievalQAWithSourcesChain

from askhr.search.hybrid_retriever import persist_bm25_index
from askhr.search.qa_chain import build_qa_chain, generate_answer as _generate_answer


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

        # See _qa_chain property below for why this is cached rather
        # than rebuilt inside generate_answer().
        self._qa_chain: Optional[RetrievalQAWithSourcesChain] = None

    @property
    def llm(self) -> Any | RunnableWithFallbacks[PromptValue | str | Sequence[Any], AIMessage]:
        if self._llm is None:
            # build_llm_with_fallback() chains Groq/Qwen (primary) ->
            # HuggingFace Gemma -> HuggingFace Mistral, per
            # core/config.py's three model configs. See
            # core/llm_wrappers.py for the fallback + logging design.
            self._llm = build_llm_with_fallback()
        return self._llm

    @property
    def qa_chain(self) -> RetrievalQAWithSourcesChain:
        """
        Lazily builds and CACHES the retrieval+generation chain, instead
        of rebuilding it on every generate_answer() call.

        Before this caching was added, every single query rebuilt the
        entire retrieval pipeline from scratch via build_qa_chain() ->
        build_reranking_retriever() -> build_hybrid_retriever(), which
        meant, per query:
          - vsm.get_all_documents() pulled EVERY chunk in the corpus
            back out of Chroma,
          - BM25Retriever.from_documents() re-tokenized that entire
            corpus into a brand new BM25 index,
          - HuggingFaceCrossEncoder(...) reconstructed the cross-encoder
            wrapper object.
        None of that changes between queries — only ingest_documents()
        (a full re-index) can invalidate it, so the cost of building it
        should be paid once per index, not once per question. This
        matters more as the policy corpus grows: rebuilding a BM25 index
        over the whole corpus on every query scales with corpus size and
        sits directly on the user-facing request path.

        _qa_chain is deliberately reset to None inside ingest_documents()
        below, so a re-index always gets a freshly-built chain that
        reflects the new corpus — this cache is invalidated on writes,
        not time-based or size-based.

        Concurrency note: this is a plain instance attribute with no
        lock. In this project's current single-worker Streamlit usage
        that's fine; if this pipeline is later served from multiple
        threads/requests concurrently (e.g. behind an API), a re-index
        racing with an in-flight query against the old chain would need
        a lock or a swap-then-publish pattern — not needed yet, but
        worth remembering before that day comes.
        """
        if self._qa_chain is None:
            self._qa_chain = build_qa_chain(self.llm, self._vsm)
        return self._qa_chain

    def ingest_documents(self) -> Iterator[str]:
        """
        Loads, splits, and stores the company policy documents from
        settings.policies_dir. Yields progress messages (same generator
        pattern as the old process_urls()).
        """
        yield "Resetting vector store...✅"
        self._vsm.reset()

        # Invalidate the cached chain BEFORE re-ingesting starts, not
        # after. If ingestion raises partway through (e.g. a malformed
        # PDF), we want the next generate_answer() call to fail loudly
        # (chain is None -> vsm.store won't have current data anyway)
        # rather than silently keep serving the stale pre-reset chain.
        self._qa_chain = None

        yield "Loading policy documents...✅"
        data = load_documents()

        yield "Splitting text into chunks...✅"
        docs = split_documents(data)

        yield "Building keyword search (BM25) index...✅"
        persist_bm25_index(docs, settings.bm25_index_path)

        yield "Adding chunks to vector database...✅"
        self._vsm.add_documents(docs)

        yield "Done adding docs to vector database...✅"

    def generate_answer(self, query: str) -> Tuple[str, List[Citation]]:
        if self._vsm.store is None:
            raise RuntimeError("Vector database is not initialized")

        return _generate_answer(self.qa_chain, self.llm, query)


if __name__ == "__main__":
    # Manual smoke test: ingests the real policy documents, asks one
    # question, prints the answer + citations. Needs GROQ_API_KEY (and
    # whichever HuggingFace/featherless-ai key the fallback models need)
    # set in .env. Run with: python -m core.pipeline (from src/, with the
    # venv active) — confirms ingestion + retrieval + generation +
    # citations + fallback wiring all work together before you commit.
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