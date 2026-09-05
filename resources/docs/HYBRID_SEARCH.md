# Hybrid Search

What hybrid retrieval is, why it's needed, its objective, and exactly
how it works in this project — covering `src/askhr/search/hybrid_retriever.py`
and its role in the broader retrieval pipeline (`qa_chain.py`,
`reranker.py`).

## Objective

Retrieval's job is to find, from the full set of indexed policy chunks,
the ones actually relevant to a user's question — cheaply enough to run
on every query. Hybrid search exists because **no single retrieval
strategy is good at everything**: some queries hinge on exact wording,
others on meaning that's expressed differently than the source text.
Combining two complementary strategies covers both cases without
sacrificing either.

## The problem it solves

- **Vector (semantic) search alone**: embeds the query and every chunk
  into vectors, ranks by similarity. Excellent for paraphrased questions
  ("what if I'm sick?" matching a chunk that says "unwell" and "time
  away from work"). Weak on exact terms — a specific code, form number,
  or precise phrase may not be the closest semantic match to anything,
  especially if surrounding words dilute the embedding.
- **BM25 (keyword) search alone**: pure term-frequency statistics, no
  meaning involved. Excellent at exact/near-exact matches. Blind to
  paraphrasing — if the query and the relevant chunk share almost no
  literal words, BM25 won't find it at all, regardless of how obviously
  related they are conceptually.

Relying on just one leaves a real gap for the other's queries. Hybrid
search runs both, on every query, and combines the results.

## How it works

### Both retrievers run on every query — unconditionally
There's no branching logic that inspects a query and decides "this looks
like a keyword search, use BM25" vs. "this looks conceptual, use vector
search." `build_hybrid_retriever()` wires both into a single
`EnsembleRetriever` (from `langchain_classic.retrievers`), which calls
**both** retrievers on **every single query** and merges their results.

### Combining the results — weighted Reciprocal Rank Fusion (RRF)
Each retriever independently returns its own ranked list. RRF combines
them purely by rank position, not by any query-type judgment: for every
chunk, at every rank position `r` it appears at (1st, 2nd, 3rd...) in
either list:

```
score += weight / (r + 60)
```

(`60` is LangChain's fixed RRF constant.) If a chunk appears in **both**
lists, its scores from each **add together** — a chunk ranked #1 by both
retrievers gets a strong combined boost. A chunk only one retriever
found still competes, weighted by that retriever's configured importance
and how highly it ranked.

### Where BM25's index comes from

Unlike Chroma, BM25 has no on-disk index format of its own — LangChain's
`BM25Retriever` is a pure in-memory wrapper around `rank_bm25`, which
needs the full text corpus available to build its term-frequency
statistics.

**This used to be rebuilt from Chroma on every cold start.**
`build_hybrid_retriever()` would call
`VectorStoreManager.get_all_documents()` — reading every chunk currently
stored in Chroma back out — and build a fresh in-memory `BM25Retriever`
from that list. Combined with `RAGPipeline`'s chain caching (see
`ARCHITECTURE.md`), this meant the cost was paid once per process
lifetime rather than once per query — but that "once" still happened on
whichever user's request triggered the first build after every app
restart, and it scaled with corpus size (more text to tokenize and index
= longer rebuild).

**Now, the index is built once during ingestion and persisted to disk.**
`ingest_documents()` (in `core/pipeline.py`) calls
`persist_bm25_index(docs, settings.bm25_index_path)` right after
chunking — using the exact same in-memory `docs` list that's about to be
embedded into Chroma, with no need to read anything back out of the
vector store. That pickled index is then loaded by
`build_hybrid_retriever()` on the next cold start, via `_load_bm25_index()`:

```python
bm25_retriever = _load_bm25_index(settings.bm25_index_path)
if bm25_retriever is None:
    # fallback: no persisted index found (e.g. first-ever run before
    # any ingestion, or a corrupted/incompatible pickle) — rebuild the
    # old way, from whatever's currently in Chroma
    documents = vsm.get_all_documents()
    bm25_retriever = _build_bm25_retriever(documents)
```

**What this does and doesn't solve.** Loading a pickle is dominated by
disk I/O + deserialization, not the tokenize/index-build computation
that dominates a from-scratch rebuild — so this meaningfully shrinks the
cold-start cost. It does **not** eliminate cold start entirely (the very
first load after any restart still reads and deserializes the pickle,
and cross-encoder/embedding model loading elsewhere in the chain has
its own, smaller first-use cost). It also does **not** address two
separate concerns that only matter at larger scale: `rank_bm25` still
holds the entire index in memory (pickling changes how fast it loads,
not how much RAM it occupies once loaded), and it still scores every
query via a linear scan across the whole corpus regardless of how the
index was built. If either of those becomes a real bottleneck — a much
larger corpus, or multiple serving replicas each duplicating the same
in-memory index — the next step is migrating keyword search to a
genuinely disk-backed, sub-linear index (Postgres full-text search or a
dedicated search engine like Elasticsearch/OpenSearch), not further
tuning of the pickle approach.

**Overwriting behavior:** a re-ingestion always overwrites the pickle
with a fresh index built from the newly-ingested corpus — there's no
scenario where a re-index leaves a stale pickle behind, since
`persist_bm25_index()` runs unconditionally as part of
`ingest_documents()`, before the old data is even replaced in Chroma.

### Configuration
```python
bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.5"))
vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))
```
Equal weighting is a neutral starting point, not a tuned value — the
faithfulness evaluation script (`eval/evaluate_faithfulness.py`) is what
should actually inform shifting this.

### How this fits into the broader retrieval pipeline
`build_hybrid_retriever()` is called from two places:
- Directly, if you just want hybrid retrieval on its own.
- Wrapped inside `search/reranker.py`'s `build_reranking_retriever()`,
  which is what `qa_chain.py` actually uses — hybrid retrieval fetches a
  **wide** candidate pool (`settings.rerank_candidate_k`, default 15),
  then a cross-encoder re-ranks and narrows it down to `settings.top_k`
  before reaching the LLM. See `RE_RANKER.md` for that second stage.

## How it helps, concretely

Two example queries make the contrast clear:

- **"What does reference code LV-2024-07 apply to?"** — a rare, specific
  token. BM25 alone finds the matching chunk immediately, ranked #1.
  Vector search with generic embeddings may not rank it as highly, since
  a rare code carries little semantic "meaning" for an embedding model
  to latch onto.
- **"What happens if I'm sick and need days off?"** — deliberately
  paraphrased, sharing almost no exact words with a document phrased as
  "staff members may take time away from work when unwell." BM25 alone
  can miss this chunk entirely. Vector search finds it correctly via
  meaning, not literal overlap.

Hybrid search handles both without needing to know in advance which kind
of question is coming.

## Testing this module

```
uv run python -m askhr.search.hybrid_retriever
```

Demonstrates both query types side by side — BM25 alone, vector alone,
and the fused hybrid result — using the real configured embedding model
(needs network access on first run, to download it).

**Note**: this temporarily resets your real vector store to load 5
throwaway sample documents for the demo, then resets it again afterward
— re-ingest your real policy documents afterward.
