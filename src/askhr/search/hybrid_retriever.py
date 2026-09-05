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

BM25 INDEX PERSISTENCE — how the "cold start" cost is minimized:
BM25Retriever needs the full text corpus in memory to build its keyword
index (unlike Chroma, it has no on-disk index format of its own).
Previously, this index was rebuilt from scratch on every cold start
(first query after a restart) by fetching every chunk back out of
Chroma via vsm.get_all_documents() and re-tokenizing the whole corpus.

Now, the index is built ONCE during ingestion (see
core/pipeline.py's ingest_documents(), which calls persist_bm25_index()
right after chunking — using the SAME documents list that's about to be
embedded and stored in Chroma, no round-trip needed) and pickled to
settings.bm25_index_path. build_hybrid_retriever() below tries to load
that pickle first; rebuilding from Chroma only happens as a FALLBACK,
for cases like the very first run before any ingestion has happened, or
if the pickle file is missing/corrupted.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from askhr.core.config import settings
from askhr.processing.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _build_bm25_retriever(documents: List[Document]) -> BM25Retriever:
    """
    Builds a BM25Retriever from an in-memory list of Documents. This is
    the one place the actual tokenizing/indexing happens — factored out
    so both the ingestion-time build (persist_bm25_index) and the
    retrieval-time fallback (build_hybrid_retriever, when no pickle
    exists yet) share the exact same construction logic.
    """
    return BM25Retriever.from_documents(documents)


def persist_bm25_index(documents: List[Document], path: Path) -> None:
    """
    Builds a BM25 index from `documents` and pickles it to `path`.

    Called from core/pipeline.py's ingest_documents(), right after
    chunking and BEFORE (or alongside) storing those same chunks in
    Chroma — so this reuses the in-memory chunks list directly, with no
    need to read anything back out of the vector store.

    Overwrites any previously-persisted index at `path`, so a re-ingest
    always leaves behind a pickle that matches the newly-ingested
    corpus, never a stale one from a previous run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    bm25_retriever = _build_bm25_retriever(documents)

    with open(path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    logger.info("Persisted BM25 index (%d documents) to %s", len(documents), path)


def _load_bm25_index(path: Path) -> Optional[BM25Retriever]:
    """
    Attempts to load a previously-persisted BM25 index from `path`.

    Returns None (rather than raising) whenever loading isn't possible —
    the file doesn't exist yet (e.g. no ingestion has ever run in this
    environment), or the pickle is corrupted/unreadable (e.g. it was
    built with a different, incompatible library version — pickles are
    tied to the exact class definitions that existed when they were
    written). Callers are expected to fall back to rebuilding from
    Chroma in either case, rather than crashing.
    """
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Broad except is deliberate here: pickle can fail in several
        # different ways (corrupted file, version mismatch between the
        # environment that wrote it and this one, etc.), and every one
        # of them should be treated identically — log it, return None,
        # let the caller fall back to rebuilding. A cache that can't be
        # trusted should never be allowed to crash the request.
        logger.warning(
            "Failed to load persisted BM25 index from %s — will rebuild "
            "from the vector store instead.",
            path,
            exc_info=True,
        )
        return None


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

    BM25's index is loaded from the pickle written during the last
    ingestion (see module docstring above). If no pickle exists yet —
    e.g. this environment has never run ingestion — this falls back to
    the slower path of rebuilding it from vsm.get_all_documents(), same
    as before this change, so retrieval still works even without a
    persisted index; it's just not the fast path.
    """
    k = k or settings.top_k

    bm25_retriever = _load_bm25_index(settings.bm25_index_path)
    if bm25_retriever is None:
        logger.info(
            "No persisted BM25 index at %s — building it from the vector "
            "store now (slower; this should only happen once per "
            "environment, before the first ingestion writes the pickle).",
            settings.bm25_index_path,
        )
        documents = vsm.get_all_documents()
        bm25_retriever = _build_bm25_retriever(documents)

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
    #   uv run python -m askhr.search.hybrid_retriever
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