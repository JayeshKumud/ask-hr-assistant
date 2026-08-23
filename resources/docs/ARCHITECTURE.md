# Architecture

## Overview

A layered pipeline: each layer has one job, depends only on the layer(s)
below it, and is independently swappable. `core.pipeline.RAGPipeline` is
the only piece that knows about every layer — everything else stays
narrowly scoped to its own concern.

```
+---------------------------------------------------+
|                  src/gui/main.py                  |
|          (Streamlit app -- the only entry point)  |
+-----------------------+---------------------------+
                         | uses
                         v
+----------------------------------------------------+
|              core.pipeline.RAGPipeline             |
|  Orchestrates the full flow: ingestion, retrieval, |
|  generation, citation enforcement.                 |
+------+-------------------+------------------+------+
       |                   |                  |
       v                   v                  v
+--------------+   +------------------+   +---------------------+
|  ingestion   |   |   processing     |   |     search          |
|              |   |                  |   |                     |
| - load PDFs  |-->| - build embed-   |-->| - hybrid retrieval  |
| - split into |   |   ding function  |   |   (BM25 + vector)   |
|   chunks,    |   | - manage the     |   | - cross-encoder     |
|   tracking   |   |   Chroma vector  |   |   re-ranking        |
|   chunk_index|   |   store          |   | - citation building |
|              |   |                  |   | - citation          |
|              |   |                  |   |   enforcement       |
+--------------+   +------------------+   +---------------------+
                         ^                        ^
                         | constants               | versioned
                +--------+---------+       +--------+--------+
                |  core/config.py  |       | config/prompts  |
                | (env-driven,     |       | .yaml           |
                |  root-anchored)  |       +-----------------+
                +------------------+

+----------------------------------------------------+
|                     src/eval/                      |
|  Offline: loads golden question CSVs, drives the   |
|  same RAGPipeline, scores faithfulness + refusal   |
|  correctness via DeepEval. Independent of the GUI. |
+----------------------------------------------------+
```

## Module responsibilities

### `src/core/config.py`
Single source of truth for every tunable constant: model names, chunk
size, retrieval/re-ranking weights, token limits, file paths. Reads from
environment variables (via `.env`) with defaults. Also defines
`PROJECT_ROOT` — computed from this file's own location on disk
(`Path(__file__).resolve().parent.parent.parent`), NOT the process's
working directory, and used to anchor `policies_dir` and
`vector_store_dir`. This matters because relative paths resolve
differently depending on how Python was launched (`python -m core.pipeline`
from the project root vs. an IDE's "Run" button, which often sets cwd to
the script's own folder) — anchoring to `__file__` makes path resolution
launcher-independent.

### `src/core/prompt_registry.py`
Loads and validates `config/prompts.yaml`. Pure config loading — doesn't
build any LangChain objects itself; `search/prompts.py` does that using
this module's `get_prompt(name)` as its source of truth.

### `src/core/pipeline.py`
`RAGPipeline` — the orchestrator. Exposes exactly two operations:
- `ingest_documents()` — load → chunk → embed → store, yielding progress
  messages as it goes (a generator, so the GUI can show live status).
- `generate_answer(query)` — retrieve → re-rank → generate → enforce
  citations → return `(answer, citations)`.

Initializes the LLM client and vector store **lazily** (built once,
cached on the instance) rather than via module-level globals — this is
what makes `st.cache_resource` in the GUI layer work cleanly, and what
makes the eval script able to construct its own independent pipeline
instance without interference.

### `src/ingestion/`
- **`document_loader.py`**: loads every `.pdf` under `settings.policies_dir`
  via LangChain's `DirectoryLoader` + `PyPDFLoader` (one Document per
  page, `page`/`page_label` metadata set — the granularity citations
  need). Normalizes `source` metadata to just the filename, not the full
  local disk path.
- **`text_splitter.py`**: splits loaded documents into chunks
  (`RecursiveCharacterTextSplitter`, sized by `settings.chunk_size`/
  `chunk_overlap`), and adds a `chunk_index` to each chunk's metadata —
  its position among chunks from the same page. Combined with page
  number, this is what lets a citation point at roughly a specific
  paragraph rather than just "somewhere on this page".

### `src/processing/`
- **`embeddings.py`**: factory for the configured HuggingFace embedding
  function (`settings.embedding_model`).
- **`vector_store.py`**: `VectorStoreManager` owns the Chroma collection
  lifecycle — lazy init, reset, add documents (each gets a fresh UUID),
  `as_retriever(k=...)` for vector search, and `get_all_documents()`,
  which reconstructs every stored chunk back into `Document` objects.
  That last method exists specifically because BM25 has no persisted
  index of its own — it needs the full corpus in memory, rebuilt from
  whatever's actually in Chroma right now, so it can never drift out of
  sync after a re-ingestion.

### `src/search/`
- **`prompts.py`**: builds `PROMPT`, `EXAMPLE_PROMPT`, and
  `VERIFICATION_PROMPT` (LangChain `PromptTemplate` objects) from
  `config/prompts.yaml` via `prompt_registry.py`. `PROMPT` is a persona
  line prepended to LangChain's built-in `stuff_prompt.template`;
  `EXAMPLE_PROMPT` formats one retrieved chunk; `VERIFICATION_PROMPT` is
  used only by `citation_enforcer.py`.
- **`citations.py`**: `Citation` dataclass (source, page, page_label,
  chunk_index, snippet) + `extract_citations()`, which converts raw
  retrieved chunks into a deduplicated citation list, keyed on
  `(source, page, chunk_index)`.
- **`hybrid_retriever.py`**: `build_hybrid_retriever()` combines BM25
  keyword search and Chroma vector search via LangChain's
  `EnsembleRetriever` (weighted Reciprocal Rank Fusion). See
  `HYBRID_SEARCH.md` for the full mechanism.
- **`reranker.py`**: `build_reranking_retriever()` wraps the hybrid
  retriever in a `ContextualCompressionRetriever`, adding a cross-encoder
  re-ranking pass (`CrossEncoderReranker` + `HuggingFaceCrossEncoder`)
  that narrows a wide candidate pool down to the most relevant few
  chunks. See `RE_RANKER.md`.
- **`citation_enforcer.py`**: `enforce_citations()` — a second, focused
  LLM call that checks whether a generated answer's claims are actually
  supported by the retrieved excerpts (via `VERIFICATION_PROMPT`), and
  replaces the answer with an explicit refusal if not. Two distinct
  refusal messages: one for "nothing relevant was retrieved at all", one
  for "something was retrieved, but the answer isn't clearly supported
  by it" (the latter still returns the citations found, for transparency).
- **`qa_chain.py`**: ties it all together — `build_qa_chain()` constructs
  the `RetrievalQAWithSourcesChain` using the re-ranking retriever and
  the two main prompts; `generate_answer()` invokes the chain, extracts
  citations, and runs them through enforcement before returning.

### `src/eval/`
Independent of the GUI — a separate, offline evaluation path:
- **`dataset_loader.py`**: loads the two golden question CSVs
  (`resources/sample_questions/`) into typed dataclasses.
- **`groq_deepeval_llm.py`**: adapts the project's `ChatGroq` instance to
  DeepEval's `DeepEvalBaseLLM` interface, so DeepEval's metrics (which
  default to OpenAI) can run on Groq instead.
- **`evaluate_faithfulness.py`**: runs all 20 positive questions through
  the real pipeline, scores each via DeepEval's `FaithfulnessMetric`, and
  runs the 5 negative (out-of-scope) questions checking for correct
  refusal. Prints a report, exits non-zero on failure — intended as the
  eventual CI gate (not wired up yet).

### `src/gui/main.py`
The app's only entry point. Wraps a single cached `RAGPipeline` instance
(`st.cache_resource`) so the vector store survives Streamlit's
rerun-the-whole-script-on-every-interaction model. One button
("Index Policy Documents"), one text input, citations rendered as
expandable sections showing doc + page + backing text.

## Key design decisions

- **`src` layout, packages resolved from `src/`** (`where = ["src"]` in
  `pyproject.toml`): `core`, `ingestion`, `processing`, `search`, `eval`,
  and `gui` are top-level importable packages once installed
  (`pip install -e .`), not nested under a `src.` prefix.

- **Lazy initialization over module globals**: `RAGPipeline` builds the
  LLM client and vector store on first access, cached on the instance,
  rather than mutating module-level globals via `global`.

- **venv + `pyproject.toml` only**: no conda, no `environment.yaml`, no
  separate `requirements.txt`. One dependency list, one install command.

- **PDF-only ingestion**: HTML support existed briefly but was dropped —
  when both a `.pdf` and `.html` version of the same document existed,
  they'd get indexed twice unless explicitly deduplicated, and PDF gives
  page-level metadata HTML doesn't. Simpler to support one format well.

- **Citations built from retrieved metadata, not the LLM's own
  "SOURCES:" output**: the LLM's self-reported sources were unreliable
  (repeated the same filename multiple times, no page info). Citations
  are constructed in code from `result["source_documents"]` instead —
  what the retriever actually returned, not what the LLM claims it used.

- **Citation enforcement as a second LLM call, not a heuristic**: word-
  overlap or keyword-matching between answer and context is weak and
  easy to fool. A dedicated verification prompt, asking the LLM directly
  whether the answer's claims appear in the excerpts, is more reliable —
  at the cost of doubling LLM calls per query.

- **Prompts in versioned YAML, not Python strings**: prompt wording is
  part of this system's behavior, not incidental code — worth the same
  change-tracking as anything else that affects what the model does.

- **DeepEval instead of RAGAS**: RAGAS is currently broken as a pip
  dependency across every version tested (unconditional import of a
  removed `langchain_community` path — a confirmed, unresolved upstream
  bug). DeepEval was verified to install and import cleanly against this
  project's actual dependency versions before being adopted.
