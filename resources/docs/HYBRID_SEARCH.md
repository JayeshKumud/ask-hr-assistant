# Hybrid Search

This document covers everything under `src/search/`: what each file does,
how they fit together, and — in detail — how the BM25 + vector hybrid
retrieval mechanism actually works under the hood.

## The files

### `src/search/prompts.py`
Defines the two prompt templates the QA chain uses:

- **`EXAMPLE_PROMPT`** formats a single retrieved chunk before it's
  inserted into the main prompt: `"Content: {page_content}\nSource: {source}"`.
  Its `input_variables` must match keys that actually exist on every
  retrieved chunk's metadata — `source` comes from
  `ingestion/document_loader.py`. Add a variable here that isn't present
  on every chunk and you get a `KeyError` at query time, not at startup.
- **`PROMPT`** is the outer instruction template sent to the LLM: a
  one-line persona ("You are a helpful assistant for NexaCore's employee
  leave and visa policy...") prepended to LangChain's built-in
  `stuff_prompt.template`, which defines the
  `"FINAL ANSWER: ...\nSOURCES: ..."` output format.

Note: as of Phase 2, the model's own self-reported `SOURCES:` line isn't
actually trusted for citations — see `citations.py` below. The persona
half of `PROMPT` still matters for tone; the `SOURCES` half is mostly
vestigial now. Phase 6 will move both templates out of Python source
entirely into a versioned config file.

### `src/search/citations.py`
Defines `Citation` — a small dataclass holding exactly which document,
page, and chunk a piece of the answer came from, plus the actual backing
text (`snippet`). `extract_citations(source_documents)` converts the raw
chunks a retriever returned into a deduplicated list of these, keyed on
`(source, page, chunk_index)` so the same location never gets listed
twice even if retrieval returned overlapping chunks.

This is deliberately independent of what the LLM says about its own
sources — it's built straight from the retriever's actual output, which
is more trustworthy. Phase 5 (citation enforcement) will reuse
`extract_citations()` to check whether an answer's claims are actually
backed by these chunks before returning it.

### `src/search/hybrid_retriever.py`
Builds the combined BM25 + vector retriever — the subject of the rest of
this document. One function, `build_hybrid_retriever(vsm)`, does the
work; see **How hybrid search works** below for the mechanism.

### `src/search/qa_chain.py`
Ties everything together into the actual chain LangChain runs:

- `build_qa_chain(llm, vsm)` constructs a `RetrievalQAWithSourcesChain`
  using `hybrid_retriever.build_hybrid_retriever(vsm)` as its retriever,
  and the two templates from `prompts.py` to format the prompt.
- `generate_answer(chain, query)` invokes the chain and returns
  `(answer, citations)` — the citations built via `citations.py` from
  whatever chunks the hybrid retriever actually returned.

`core/pipeline.py`'s `RAGPipeline.generate_answer()` is the only caller
of this module — it's the last stop before the answer reaches the GUI.

## How hybrid search works

### The core idea
Vector (semantic) search embeds the query and finds chunks whose meaning
is closest — great for paraphrased questions, weak on exact terms that
don't carry much semantic weight (a specific code, a form number, an
exact phrase). BM25 is the opposite: pure keyword/term-frequency
matching, no embeddings involved — great for exact terms, blind to
paraphrasing. Hybrid search runs both and combines their results so
neither weakness is fatal on its own.

### Both retrievers run on every query — always
This is the most common misunderstanding: there's no branching logic
anywhere that inspects a query and decides "this looks like a keyword
search, use BM25" versus "this looks conceptual, use vector search."
`EnsembleRetriever` (from `langchain_classic.retrievers`) calls **both**
retrievers on **every single query**, unconditionally. Each one
independently returns its own ranked list of `settings.top_k` chunks.

### Combining the two ranked lists — Reciprocal Rank Fusion (RRF)
The two ranked lists then get merged using weighted RRF. For every chunk,
at every rank position `r` it appears at in either list (1st, 2nd, 3rd...):

```
score += weight / (r + 60)
```

(`60` is LangChain's fixed RRF constant, `c`, in `EnsembleRetriever`.)
If the same chunk appears in **both** lists, its scores from each list
**add together**. Everything is then re-sorted by this combined score.

Concretely:

- A chunk ranked **#1 by both** retrievers gets a strong combined boost
  — both contributions stack.
- A chunk **only BM25 found**, ranked #1 there, still competes — its
  score is `bm25_weight / 61`, weighted purely by how important you've
  configured BM25 to be.
- A chunk that ranks low in both lists (say, 5th in each) contributes
  very little either way — `weight / 65` is a small number.

Nothing here is a judgment call about query type — it's pure
rank-position arithmetic, weighted by two configured numbers.

### Configuration
Two settings in `core/config.py` control the balance:

```python
import os

bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.5"))
vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))
```

Equal weighting (`0.5` / `0.5`) is a neutral starting point, not a tuned
value — Phase 6's evaluation script (faithfulness scoring against the
golden question set) is what should actually inform whether to shift
this toward one side.

### Where BM25's index comes from
Unlike Chroma, BM25 has no persistent on-disk index in this system.
`build_hybrid_retriever()` calls `vsm.get_all_documents()` — which reads
every chunk currently stored in Chroma back out and reconstructs it as a
`Document` — then builds a fresh in-memory `BM25Retriever` from that list
on every call. This means BM25 always reflects whatever's *actually* in
the vector store right now (it can never drift out of sync after a
re-ingestion), at the cost of rebuilding the index on every query. For a
handful of short policy PDFs, that cost is negligible; it would be worth
caching if the corpus grew much larger.

### Seeing it in action
`hybrid_retriever.py`'s `if __name__ == "__main__":` block demonstrates
this directly — it runs two contrasting example queries (one
keyword-heavy, one paraphrased with no exact keyword overlap) through
BM25 alone, vector alone, and the fused hybrid retriever, printing all
three side by side. Run it with:

```
python -m search.hybrid_retriever
```

**Note:** this temporarily resets your real vector store to load 5
throwaway sample documents for the demo, then resets it again afterward
— you'll need to re-ingest your real policy documents (via
`ingest_documents()` or the GUI's "Index Policy Documents" button)
after running it.