# Cross-Encoder Re-Ranking

What re-ranking is, why hybrid retrieval's own ranking isn't enough on
its own, its objective, and how it helps — covering
`src/askhr/search/reranker.py` and how it builds on hybrid retrieval.

## Objective

Retrieval (hybrid search — see `HYBRID_SEARCH.md`) casts a wide net
*cheaply*: it fetches a candidate pool of plausibly-relevant chunks fast,
across an entire corpus. Re-ranking's job is different: take that
already-narrowed pool and pick the best few from it *accurately*, using
a technique too slow to run over the whole corpus but well worth
applying to a small candidate set.

## Why hybrid retrieval's ranking isn't enough on its own

Hybrid retrieval already produces a ranked list (via Reciprocal Rank
Fusion of BM25 and vector scores) — so why rank it again?

Because **both BM25 and vector search score the query and each chunk
independently, then compare the results after the fact** — neither ever
looks at the query and a specific chunk *together*:

- **BM25** scores on term-frequency statistics alone — no concept of
  meaning. It can rank a perfectly relevant chunk near the bottom simply
  because the wording differs from the query.
- **Vector search** is a **bi-encoder**: it embeds the query and each
  chunk *separately*, then compares the two resulting vectors. The query
  and document never interact until that final similarity comparison —
  a lot of nuance can get lost when meaning is compressed into one fixed
  vector before any comparison happens.
- **RRF fusion** compounds this: it only considers *rank position*, not
  the underlying relevance strength. A strong match and a mediocre match
  can end up with identical fusion scores if they happened to land at
  the same rank in their respective lists.

A **cross-encoder** works differently: it feeds the query and a
candidate chunk into the model **together, concatenated, as one input**.
The model's attention layers directly compare specific words and phrases
between the two while computing a single relevance score — not two
independently-computed summaries compared afterward. This is
consistently more accurate than either BM25 or bi-encoder vector search.

The catch is cost: a full model forward pass per `(query, chunk)` pair
doesn't scale to an entire corpus. That's exactly why it's used to
re-rank a small pool (`settings.rerank_candidate_k = 15`) rather than
search a full index directly — hybrid retrieval's job is getting the
right chunk *somewhere* into that pool of 15; re-ranking's job is
correctly identifying which one it is and putting it first.

## Alternatives considered

- **Cohere Rerank / Voyage AI Rerank / Jina Reranker** — managed, paid,
  cloud-hosted reranking APIs. Strong quality, but an external paid
  dependency and network round-trip per query. Ruled out in favor of
  staying self-contained with this project's local, open-source stack.
- **ColBERT (late-interaction retrieval)** — a middle ground: encodes
  query and document tokens separately (so it *can* scale to full-corpus
  search, unlike a cross-encoder) but delays interaction to a finer
  per-token comparison. More accurate than a plain bi-encoder, but needs
  a specialized index and more infrastructure than a handful of policy
  PDFs warrants.
- **MMR (Maximal Marginal Relevance)** — solves a *different* problem:
  result *diversity* (avoiding near-duplicate chunks), not relevance
  precision. Not a substitute for cross-encoder scoring.

Cross-encoder re-ranking via `sentence-transformers` was chosen: local
(no API key, no per-call cost, no ongoing network dependency once the
model's cached), well-established for this exact task, and integrates
directly into LangChain's existing retriever abstractions.

## How it works — the three LangChain pieces

`build_reranking_retriever()` wires together:

1. **`HuggingFaceCrossEncoder`** (`langchain_community.cross_encoders`) —
   wraps a `sentence-transformers` cross-encoder model
   (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default,
   `settings.reranker_model`). Its `.score()` takes a list of
   `(query, chunk_text)` pairs, returns one relevance score per pair.
2. **`CrossEncoderReranker`** (`langchain_classic.retrievers.document_compressors`)
   — a "document compressor": given a query and candidate `Document`s,
   scores every pair via the cross-encoder, sorts descending, keeps only
   the top `top_n` (set to `settings.top_k`).
3. **`ContextualCompressionRetriever`** (`langchain_classic.retrievers`)
   — wraps a `base_retriever` and `base_compressor` together. On
   `.invoke(query)`: calls the base retriever (hybrid retrieval, fetching
   `settings.rerank_candidate_k` candidates), passes the result to the
   compressor (the cross-encoder reranker), returns the narrowed result.
   From the outside it behaves like any other retriever —
   `qa_chain.py`'s `build_qa_chain()` uses it exactly where the plain
   hybrid retriever was used before re-ranking was added, with no other
   code needing to know re-ranking is happening.

## Function reference

### `build_reranking_retriever(vsm: VectorStoreManager) -> ContextualCompressionRetriever`
The only function in `reranker.py`. Builds and returns the full
retrieve-then-rerank pipeline, ready to hand to
`RetrievalQAWithSourcesChain` as its retriever.

### `FakeCrossEncoder.score()` (test-only, inside `if __name__ == "__main__":`)
Not part of the real pipeline — `HuggingFaceCrossEncoder` always
downloads its real model in `__init__`, with no way to inject a
substitute, so this hand-assigns scores for a small fixed set of example
sentences to let the demo prove the retrieve-then-rerank *wiring* works
correctly offline.

## How it helps, concretely

The offline demo (below) shows a candidate pool where the genuinely
best-matching chunk starts in **3rd place** — simulating what hybrid
retrieval's rank-position fusion might produce — and the cross-encoder
correctly promotes it to **1st**, while an irrelevant chunk drops out of
the top results entirely. That reordering, based on actually reading the
query and each candidate together rather than combining two independent
rankings, is the entire value re-ranking adds.

## Testing this module

```
uv run python -m askhr.search.reranker
```

Runs instantly, fully offline (`FakeCrossEncoder`) — confirms the
retrieve → compress → return wiring.

To see the **real** cross-encoder model, run `uv run python -m askhr.core.pipeline`
instead — exercises `build_reranking_retriever()` for real via
`qa_chain.py`, downloading `cross-encoder/ms-marco-MiniLM-L-6-v2` on
first use.
