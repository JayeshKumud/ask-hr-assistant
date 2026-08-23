"""
Search: combines BM25 (keyword) and vector (semantic) retrieval.

Vector search alone can miss exact-term queries — a question containing a
specific phrase like "Form I-129" or "25 working days" may not be the
closest semantic match to anything in the index if the embedding model
weights the surrounding words more heavily. BM25 excels precisely at
this: exact/near-exact keyword overlap, no embedding involved. Combining
both, via LangChain's EnsembleRetriever (Reciprocal Rank Fusion), gets
the benefit of each without picking one at the expense of the other.

IMPORTANT — how the combination actually works: BOTH retrievers run on
EVERY query, unconditionally. There is no logic anywhere that inspects
the query and decides "this looks like a keyword search, use BM25" or
"this looks conceptual, use vector search". Each retriever independently
returns its own ranked list of settings.top_k documents; EnsembleRetriever
then merges those two ranked lists using weighted Reciprocal Rank Fusion:
for each document, at each rank position r it appears at (1st, 2nd...),
it earns `weight / (r + 60)` — and if it appears in both lists, those
scores ADD together. Everything is re-sorted by that combined score. A
document ranked #1 by both retrievers gets a strong combined boost; a
document only one retriever found still competes, just weighted by that
retriever's configured weight and how high it ranked.
"""
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from core.config import settings
from processing.vector_store import VectorStoreManager


from typing import Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from core.config import settings
from processing.vector_store import VectorStoreManager


def build_hybrid_retriever(vsm: VectorStoreManager, k: Optional[int] = None) -> EnsembleRetriever:
    """
    Builds an EnsembleRetriever combining BM25 and vector search over
    whatever is currently stored in the vector store.

    Args:
        vsm: The vector store manager to retrieve from.
        k: Optional override for how many documents each retriever
            returns. Defaults to settings.top_k. search/reranker.py
            passes settings.rerank_candidate_k here instead, to fetch a
            wider candidate pool for the cross-encoder to narrow down.

    BM25Retriever is rebuilt from vsm.get_all_documents() each time this
    is called, rather than persisted — BM25 has no on-disk index of its
    own in this setup, so it always reflects the current contents of
    Chroma. For a small policy-document corpus like this one, rebuilding
    it per query is cheap; if the corpus grows much larger, this would be
    worth caching instead of rebuilding every call.
    """
    k = k or settings.top_k
    documents = vsm.get_all_documents()

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    vector_retriever = vsm.as_retriever(k=k)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[settings.bm25_weight, settings.vector_weight],
    )


def _print_results(label: str, docs) -> None:
    print(f"\n{label}:")
    if not docs:
        print("  (no results)")
    for i, doc in enumerate(docs, start=1):
        print(f"  {i}. [{doc.metadata.get('source')}] {doc.page_content[:70]!r}")


if __name__ == "__main__":
    # Manual check demonstrating BOTH retrievers separately, plus the
    # fused hybrid result, for two DIFFERENT kinds of queries — one
    # keyword-heavy (BM25's strength), one paraphrased with no exact
    # keyword overlap (vector search's strength).
    #
    # This uses the REAL configured embedding model (not a fake one) —
    # it needs network access on first run to download the model from
    # HuggingFace, same as ingest_documents() does. Run with:
    #   python -m search.hybrid_retriever
    from langchain_core.documents import Document

    sample_docs = [
        Document(
            page_content=(
                "Reference code LV-2024-07 applies to legacy annual leave "
                "carryover requested before the 2024 policy update."
            ),
            metadata={"source": "leave.pdf", "page": 0, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Staff members may take time away from work when they are "
                "unwell, provided they inform their line manager in advance."
            ),
            metadata={"source": "leave.pdf", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Full-time employees receive 25 working days of annual "
                "leave per year, plus applicable public holidays."
            ),
            metadata={"source": "leave.pdf", "page": 0, "chunk_index": 1},
        ),
        Document(
            page_content=(
                "Form I-129 must be filed for H-1B visa sponsorship "
                "requests submitted by the employer."
            ),
            metadata={"source": "visa.pdf", "page": 0, "chunk_index": 0},
        ),
        Document(
            page_content=(
                "Remote work arrangements require written approval from "
                "the employee's direct manager and HR."
            ),
            metadata={"source": "remote.pdf", "page": 0, "chunk_index": 0},
        ),
    ]

    vsm = VectorStoreManager()
    vsm.reset()
    vsm.add_documents(sample_docs)

    bm25_only = BM25Retriever.from_documents(sample_docs)
    bm25_only.k = 3
    vector_only = vsm.as_retriever()
    hybrid = build_hybrid_retriever(vsm)

    # --- Query 1: exact-keyword-heavy ---
    # Shares a rare, specific token ("LV-2024-07") with exactly one
    # document. BM25 should nail this trivially; it's the kind of query
    # semantic embeddings sometimes underweight in favor of "vibes".
    keyword_query = "What does reference code LV-2024-07 apply to?"
    print("=" * 70)
    print(f"QUERY 1 (keyword-heavy): {keyword_query!r}")
    _print_results("BM25 alone", bm25_only.invoke(keyword_query))
    _print_results("Vector alone", vector_only.invoke(keyword_query))
    _print_results("Hybrid (fused)", hybrid.invoke(keyword_query))

    # --- Query 2: paraphrased, no exact keyword overlap ---
    # Deliberately shares almost no exact words with the sick-leave
    # document ("unwell", "time away" vs. "sick", "ill", "days off").
    # BM25 should struggle here since it only matches literal tokens;
    # vector search should still find it via meaning.
    semantic_query = "What happens if I'm sick and need days off?"
    print("\n" + "=" * 70)
    print(f"QUERY 2 (paraphrased/semantic): {semantic_query!r}")
    _print_results("BM25 alone", bm25_only.invoke(semantic_query))
    _print_results("Vector alone", vector_only.invoke(semantic_query))
    _print_results("Hybrid (fused)", hybrid.invoke(semantic_query))

    vsm.reset()  # leave no test data behind in the real vector store