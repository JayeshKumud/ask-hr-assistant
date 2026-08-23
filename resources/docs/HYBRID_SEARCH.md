# Hybrid Search

What hybrid retrieval is, why it's needed, its objective, and exactly
how it works in this project — covering `src/search/hybrid_retriever.py`
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
Unlike Chroma, BM25 has no on-disk persisted index in this system.
`build_hybrid_retriever()` calls `VectorStoreManager.get_all_documents()`
— which reads every chunk currently stored in Chroma back out — and
builds a fresh in-memory `BM25Retriever` from that list on every call.
This means BM25 always reflects whatever's actually in the vector store
right now; it can never silently drift out of sync after a re-ingestion.
The tradeoff is rebuilding the index on every query — negligible for a
handful of policy PDFs, worth caching if the corpus grew much larger.

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
python -m search.hybrid_retriever
```

Demonstrates both query types side by side — BM25 alone, vector alone,
and the fused hybrid result — using the real configured embedding model
(needs network access on first run, to download it).

**Note**: this temporarily resets your real vector store to load 5
throwaway sample documents for the demo, then resets it again afterward
— re-ingest your real policy documents afterward.
