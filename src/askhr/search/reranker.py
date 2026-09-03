"""
Search: re-scores hybrid retrieval's candidates with a cross-encoder for
higher-precision final ranking.

Hybrid retrieval (BM25 + vector, combined via Reciprocal Rank Fusion) is
good at CASTING A WIDE NET cheaply, but RRF's fusion is purely rank-
position arithmetic — it never actually reads the query and a candidate
chunk together. A cross-encoder does something qualitatively different:
it takes the (query, chunk) PAIR as joint input and scores how well they
actually match, which tends to be far more precise than combining two
independently-computed rankings. The tradeoff is cost: a cross-encoder is
too slow to run over an entire corpus, which is why it's used to re-rank
a small candidate pool rather than search the whole index directly.

Wired together via three LangChain pieces:
- HuggingFaceCrossEncoder wraps a sentence-transformers cross-encoder
  model (default: cross-encoder/ms-marco-MiniLM-L-6-v2 — small, fast,
  well-established for exactly this task).
- CrossEncoderReranker is a "document compressor": given a query and a
  list of candidate Documents, it scores every (query, candidate) pair
  and keeps only the top_n.
- ContextualCompressionRetriever wires a base retriever + a compressor
  together: it first calls the base retriever (hybrid retrieval, fetching
  settings.rerank_candidate_k candidates), then runs the compressor
  (cross-encoder re-ranking) on the result, narrowing down to
  settings.top_k before it's returned.
"""
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from askhr.core.config import settings
from askhr.processing.vector_store import VectorStoreManager
from askhr.search.hybrid_retriever import build_hybrid_retriever


def build_reranking_retriever(vsm: VectorStoreManager) -> ContextualCompressionRetriever:
    """
    Builds the full retrieval pipeline: hybrid retrieval (wide candidate
    pool) -> cross-encoder re-ranking (narrowed to the most relevant).

    This is what search/qa_chain.py's build_qa_chain() uses as the
    chain's retriever, replacing the plain hybrid retriever from Phase 3.
    """
    candidate_retriever = build_hybrid_retriever(vsm, k=settings.rerank_candidate_k)

    cross_encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.top_k)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=candidate_retriever,
    )


if __name__ == "__main__":
    # Manual check demonstrating what re-ranking actually changes: a
    # candidate pool where hybrid retrieval's rank-position fusion puts
    # the genuinely best-matching chunk in 3rd place (it shares fewer
    # exact words with the query, and isn't the top vector match either),
    # while a cross-encoder — reading query and chunk together — should
    # correctly promote it based on actual relevance rather than any
    # position-based heuristic.
    #
    # Uses a FAKE cross-encoder with manually-assigned scores instead of
    # the real HuggingFaceCrossEncoder, since that class always downloads
    # its model in __init__ with no way to inject a stub — this keeps the
    # test runnable offline while still proving the re-ranking WIRING
    # (ContextualCompressionRetriever: retrieve wide -> compress ->
    # return narrowed results) actually works end to end.
    #
    # To see the REAL cross-encoder model in action instead, run
    # `python -m core.pipeline` after this — that uses the real
    # HuggingFaceCrossEncoder via build_reranking_retriever() above, and
    # will download cross-encoder/ms-marco-MiniLM-L-6-v2 on first use.
    from typing import List, Tuple

    from langchain_classic.retrievers import ContextualCompressionRetriever as _CCR
    from langchain_community.cross_encoders.base import BaseCrossEncoder
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class FakeCrossEncoder(BaseCrossEncoder):
        """Deterministic fake scorer — hand-assigned relevance scores per
        snippet, so the demo's outcome doesn't depend on downloading a
        real model."""

        SCORES = {
            "Employees who are unwell should inform their manager promptly.": 0.95,
            "Remote work requires written approval from HR.": 0.10,
            "The office is closed on all recognized public holidays.": 0.85,
            "Annual performance reviews occur every March.": 0.05,
        }

        def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
            return [self.SCORES.get(doc_text, 0.0) for _, doc_text in text_pairs]

    class FakeWideRetriever(BaseRetriever):
        """Stands in for hybrid retrieval's output: a candidate pool
        already in SOME order (simulating RRF's rank fusion), not
        necessarily the best order."""

        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return [
                Document(page_content="The office is closed on all recognized public holidays."),
                Document(page_content="Remote work requires written approval from HR."),
                Document(page_content="Employees who are unwell should inform their manager promptly."),
                Document(page_content="Annual performance reviews occur every March."),
            ]

    print("Candidate order BEFORE re-ranking (simulated hybrid retrieval output):")
    for i, doc in enumerate(FakeWideRetriever()._get_relevant_documents(""), start=1):
        print(f"  {i}. {doc.page_content!r}")

    reranking_retriever = _CCR(
        base_compressor=CrossEncoderReranker(model=FakeCrossEncoder(), top_n=2),
        base_retriever=FakeWideRetriever(),
    )
    results = reranking_retriever.invoke("What should I do if I'm sick?")

    print("\nTop 2 AFTER cross-encoder re-ranking:")
    for i, doc in enumerate(results, start=1):
        print(f"  {i}. {doc.page_content!r}")