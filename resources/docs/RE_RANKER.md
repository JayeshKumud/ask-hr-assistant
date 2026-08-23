# Cross-Encoder Re-Ranking

This document covers `src/search/reranker.py`: what it does, why hybrid
retrieval alone isn't enough, what alternatives exist, and exactly how
the mechanism works.

## What it does

After hybrid retrieval (BM25 + vector search, see `HYBRIDE_SEARCH.md`)
returns a pool of candidate chunks, re-ranking takes a second pass over
that pool: it scores every `(query, chunk)` pair together using a
cross-encoder model, re-sorts by that score, and keeps only the top
`settings.top_k` — which is what actually reaches the LLM's prompt.

Retrieval's job is **casting a wide net cheaply**. Re-ranking's job is
**picking the best few from that net accurately**. They're deliberately
two separate stages because the technique that makes re-ranking accurate
(described below) is too slow to run over an entire document corpus.

## Why hybrid retrieval's ranking isn't enough on its own

It's a fair question — hybrid retrieval already produces a ranked list
(via Reciprocal Rank Fusion of BM25 and vector scores). Why re-rank a
ranking that already exists?

The answer is that **both BM25 and vector search score the query and
each chunk independently, then compare the results after the fact** —
neither ever looks at the query and a specific chunk *together*:

- **BM25** scores based on term-frequency statistics (how often query
  words appear in a chunk, adjusted for document length and term
  rarity). It has no concept of meaning at all — "sick leave" and "time
  off when unwell" share zero exact terms, so BM25 can rank a perfectly
  relevant chunk near the bottom simply because the wording differs.
- **Vector search** embeds the query into a vector and each chunk into a
  vector, *separately*, then compares those two fixed vectors with
  cosine similarity. This is called a **bi-encoder** approach — "bi"
  because the query and the document are encoded independently, in two
  separate passes, and never interact until the similarity comparison at
  the very end. A lot of nuance can get lost when meaning is compressed
  into one fixed-size vector before the comparison ever happens.
- **RRF fusion** (how hybrid retrieval combines the two) makes this
  worse in one specific way: it only looks at *rank position*, not the
  underlying relevance scores at all. A chunk that's a very strong match
  and a chunk that's a mediocre match can end up with identical fusion
  scores if they happened to land at the same rank position in their
  respective lists.

A **cross-encoder** works differently: it feeds the query and a
candidate chunk **into the model together, concatenated, as a single
input**. The model's attention layers can then directly compare specific
words and phrases between the query and the chunk while computing the
relevance score — not two independently-computed summaries compared
after the fact. This consistently produces much more accurate relevance
judgments than either BM25 or bi-encoder vector search.

The catch: doing that for every chunk in an entire corpus, for every
query, doesn't scale — you'd need one full model forward pass per
`(query, chunk)` pair, against potentially thousands of chunks. That's
why cross-encoders are used to re-rank a **small candidate pool**
(here, `settings.rerank_candidate_k = 15`) rather than search a full
index directly. Hybrid retrieval's job is to cheaply get the right chunk
*somewhere* into that pool of 15; re-ranking's job is to correctly
identify which one it is and put it first.

## Alternatives considered

Cross-encoder re-ranking isn't the only option — a few other approaches
exist, mentioned here for context on why this one was chosen:

- **Cohere Rerank / Voyage AI Rerank / Jina Reranker** — managed,
  paid, cloud-hosted reranking APIs. Often very strong quality, but adds
  an external paid dependency and a network round-trip per query. Ruled
  out here in favor of staying self-contained with the rest of this
  project's local, open-source LangChain + Chroma + HuggingFace stack.
- **ColBERT (late-interaction retrieval)** — a middle ground between
  bi-encoders and cross-encoders: it encodes query and document tokens
  separately (like a bi-encoder, so it *can* scale to full-corpus
  search) but delays the interaction to a finer per-token comparison
  step, rather than one final vector-to-vector comparison. More accurate
  than a plain bi-encoder, but requires a specialized index and more
  infrastructure than this project needs for a handful of policy PDFs.
- **MMR (Maximal Marginal Relevance) re-ranking** — a different kind of
  re-ranking entirely, optimizing for *diversity* among results (avoid
  returning five near-duplicate chunks) rather than *relevance
  precision*. Solves a different problem than the one here — worth
  knowing about, not a substitute for cross-encoder scoring.

Cross-encoder re-ranking via `sentence-transformers` was chosen because
it's local (no API key, no per-call cost, no network dependency once the
model is downloaded once), well-established for exactly this use case,
and integrates directly into LangChain's existing retriever abstractions
rather than requiring custom glue code.

## How it works — the three LangChain pieces

`build_reranking_retriever()` wires together three LangChain classes:

### 1. `HuggingFaceCrossEncoder` (from `langchain_community.cross_encoders`)
Wraps a `sentence-transformers` cross-encoder model —
`cross-encoder/ms-marco-MiniLM-L-6-v2` by default
(`settings.reranker_model`), a small, fast model well-established for
this task, trained on the MS MARCO passage ranking dataset. Its `.score()`
method takes a list of `(query, chunk_text)` pairs and returns one
relevance score per pair. The real model is downloaded from HuggingFace
the first time it's used (same one-time-download pattern as the
embedding model).

### 2. `CrossEncoderReranker` (from `langchain_classic.retrievers.document_compressors`)
A LangChain "document compressor" — given a query and a list of
candidate `Document`s, it calls the cross-encoder's `.score()` on every
`(query, chunk)` pair, sorts by score descending, and returns only the
top `top_n` (set to `settings.top_k` here). "Compressor" is LangChain's
general term for anything that takes a document list and returns a
smaller/better one — re-ranking is one specific kind of compression.

### 3. `ContextualCompressionRetriever` (from `langchain_classic.retrievers`)
Wraps a `base_retriever` and a `base_compressor` together into a single
retriever. When invoked with a query, it:
1. Calls `base_retriever.invoke(query)` — here, hybrid retrieval
   (`build_hybrid_retriever(vsm, k=settings.rerank_candidate_k)`),
   fetching a wide pool of `rerank_candidate_k` candidates.
2. Passes that pool to `base_compressor.compress_documents(...)` — here,
   the `CrossEncoderReranker` — which re-scores and narrows it to
   `top_k`.
3. Returns the narrowed, re-ranked result.

From the outside, `ContextualCompressionRetriever` behaves like any
other LangChain retriever — `search/qa_chain.py`'s `build_qa_chain()`
uses it exactly where the plain hybrid retriever was used in Phase 3,
with no other code needing to know re-ranking is happening.

## Function reference

### `build_reranking_retriever(vsm: VectorStoreManager) -> ContextualCompressionRetriever`
The only function in this module. Builds and returns the full
retrieve-then-rerank pipeline described above, ready to hand to
`RetrievalQAWithSourcesChain` as its retriever. Takes the
`VectorStoreManager` (not a raw retriever) because it needs to pass it
through to `build_hybrid_retriever()`, which in turn needs
`vsm.get_all_documents()` to build BM25's index.

### `FakeCrossEncoder.score()` (test-only, inside `if __name__ == "__main__":`)
Not part of the real pipeline — a stand-in used only by this file's
manual test, since `HuggingFaceCrossEncoder` always downloads its real
model in `__init__` with no way to inject a substitute. Returns
hand-assigned scores for a small fixed set of example sentences, so the
demo can prove the retrieve-then-rerank *wiring* works correctly offline,
without needing network access or a real model.

## Testing this module

```
python -m search.reranker
```

Runs instantly, fully offline — demonstrates a candidate pool where the
genuinely best-matching chunk starts in 3rd position (simulating what
hybrid retrieval's rank-fusion might produce) and shows the fake
cross-encoder correctly promoting it to 1st based on relevance rather
than position.

To see the **real** cross-encoder model in action (not the offline fake),
run `python -m core.pipeline` instead — that exercises
`build_reranking_retriever()` for real, via `qa_chain.py`, and will
download `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use.