# Architecture

## Overview

A layered pipeline: each layer has one job, depends only on the layer(s)
below it, and is independently swappable. `askhr.core.pipeline.RAGPipeline`
is the only piece that knows about every retrieval/generation layer;
`askhr.service.PolicyQAService` is the stable, interface-agnostic entry
point everything else (GUI today, future agents/API) is meant to depend
on instead of reaching into `RAGPipeline` directly.

```
+---------------------------------------------------+
|              src/askhr/gui/main.py                |
|          (Streamlit app -- one of possibly         |
|           several future callers)                 |
+-----------------------+---------------------------+
                         | uses
                         v
+----------------------------------------------------+
|     askhr.service.policy_qa_service.PolicyQAService |
|  Interface-agnostic facade: ask(query), reindex().  |
|  Future LangGraph agent tools call THIS, not        |
|  RAGPipeline directly.                              |
+-----------------------+----------------------------+
                         | uses
                         v
+----------------------------------------------------+
|          askhr.core.pipeline.RAGPipeline            |
|  Orchestrates the full flow: ingestion, retrieval,  |
|  generation, citation enforcement. Caches the built  |
|  retrieval chain (self._qa_chain) across queries.    |
+------+-------------------+------------------+-------+
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
                         | constants              | versioned
                +--------+---------+       +--------+--------+
                |  core/config.py  |       | config/prompts  |
                | (env-driven,     |       | .yaml           |
                |  root-anchored)  |       +-----------------+
                +------------------+

+----------------------------------------------------+
|                askhr.db (Phase 4)                   |
|  SQLAlchemy models + repository for leave balances/ |
|  requests. Independent of the RAG flow above — this |
|  is what future LangGraph agent tools (fetch leave  |
|  balance, apply leave) will read/write through.     |
|  Not yet wired to an agent; built and unit-verified. |
+----------------------------------------------------+

+----------------------------------------------------+
|                   src/askhr/eval/                   |
|  Offline: loads golden question CSVs, drives the   |
|  same RAGPipeline, scores faithfulness + refusal   |
|  correctness via DeepEval. Independent of the GUI. |
+----------------------------------------------------+
```

## Module responsibilities

### `src/askhr/core/config.py`
Single source of truth for every tunable constant: model names, chunk
size, retrieval/re-ranking weights, token limits, file paths, the BM25
index pickle path, and the database URL. Reads from environment
variables (via `.env`) with defaults. Also defines `PROJECT_ROOT` —
computed from this file's own location on disk
(`Path(__file__).resolve().parent.parent.parent.parent` — one level
deeper than before the `src/askhr/` namespace restructure, since the
file itself moved one directory deeper), NOT the process's working
directory, and used to anchor `policies_dir`, `vector_store_dir`,
`bm25_index_path`, and the default SQLite `database_url`. This matters
because relative paths resolve differently depending on how Python was
launched (`uv run python -m askhr.core.pipeline` from the project root
vs. an IDE's "Run" button, which often sets cwd to the script's own
folder) — anchoring to `__file__` makes path resolution
launcher-independent.

### `src/askhr/core/prompt_registry.py`
Loads and validates `config/prompts.yaml`. Pure config loading — doesn't
build any LangChain objects itself; `search/prompts.py` does that using
this module's `get_prompt(name)` as its source of truth.

### `src/askhr/core/pipeline.py`
`RAGPipeline` — the orchestrator. Exposes exactly two operations:
- `ingest_documents()` — resets the vector store, loads → chunks → builds
  and **persists the BM25 index to disk** (`persist_bm25_index()`, using
  the same in-memory chunk list about to be embedded — no round-trip
  through Chroma needed for this) → embeds and stores the chunks in
  Chroma, yielding progress messages as it goes (a generator, so the GUI
  can show live status). Also invalidates the cached retrieval chain
  (`self._qa_chain = None`) at the start, so the next query is guaranteed
  to rebuild against the fresh corpus.
- `generate_answer(query)` — retrieve → re-rank → generate → enforce
  citations → return `(answer, citations)`, via the cached `qa_chain`
  property (see below).

Initializes the LLM client (`self.llm`) and the retrieval chain
(`self.qa_chain`) **lazily and cache them on the instance** — built once,
reused across every subsequent call — rather than rebuilding either on
every request. Before this caching was added, `generate_answer()`
rebuilt the entire retrieval chain (hybrid retriever + cross-encoder)
from scratch on every single query; now that cost is paid once per
process (or once per re-index), not once per question. This is also
what makes `st.cache_resource` in the GUI layer work cleanly, and what
lets the eval script construct its own independent pipeline instance
without interference.

### `src/askhr/ingestion/`
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

### `src/askhr/processing/`
- **`embeddings.py`**: factory for the configured HuggingFace embedding
  function (`settings.embedding_model`).
- **`vector_store.py`**: `VectorStoreManager` owns the Chroma collection
  lifecycle — lazy init, reset, add documents (each gets a fresh UUID),
  `as_retriever(k=...)` for vector search, and `get_all_documents()`,
  which reconstructs every stored chunk back into `Document` objects.
  This method now exists as a **fallback path only** — used by
  `hybrid_retriever.py` when no persisted BM25 pickle exists yet (e.g.
  the very first run in a fresh environment, before any ingestion has
  happened), rather than as the primary way BM25's index gets built on
  every cold start.

### `src/askhr/search/`
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
  `EnsembleRetriever` (weighted Reciprocal Rank Fusion). BM25's index is
  loaded from a pickle written during the last ingestion
  (`persist_bm25_index()`/`_load_bm25_index()`), with a fallback to
  rebuilding from `vsm.get_all_documents()` if no pickle exists. See
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
  Guards against the verification LLM call returning an empty response
  (observed occasionally with certain reasoning-model configurations) by
  treating that case as "unsupported" rather than crashing.
- **`qa_chain.py`**: ties it all together — `build_qa_chain()` constructs
  the `RetrievalQAWithSourcesChain` using the re-ranking retriever and
  the two main prompts; `generate_answer()` invokes the chain, extracts
  citations, and runs them through enforcement before returning.

### `src/askhr/service/policy_qa_service.py`
`PolicyQAService` — a thin, interface-agnostic facade over `RAGPipeline`,
exposing `ask(query) -> AnswerResult` and `reindex() -> Iterator[str]`.
Added specifically so that `RAGPipeline` construction/caching isn't
owned by any one caller: before this existed, `gui/main.py` was the only
consumer and built/cached `RAGPipeline` itself via `st.cache_resource` (a
Streamlit-specific decorator). `get_policy_qa_service()` is a plain
`functools.lru_cache`-based singleton accessor instead — usable from
Streamlit, from a future LangGraph agent tool, or from a future API
endpoint, all sharing the same underlying pipeline instance within a
process without depending on each other.

### `src/askhr/db/`
Independent of the RAG flow — built for the upcoming leave-balance
agent tools, not yet wired to an agent:
- **`models.py`**: SQLAlchemy models — `LeaveBalance` (per employee, per
  leave type) and `LeaveRequest` (an audit trail of every apply-leave
  attempt, approved or rejected).
- **`session.py`**: engine + session management (`get_session()` context
  manager, commit-on-success/rollback-on-error) and `init_db()` to
  create tables. Reads only `settings.database_url` — switching from the
  default local SQLite to Postgres is a `.env` change, not a code change.
- **`repositories/leave_repository.py`**: `LeaveRepository` —
  `get_balance()`, `list_balances()`, `apply_leave()` (debits balance and
  writes an audit row on success; writes a rejected audit row and raises
  `InsufficientBalanceError` on insufficient balance, never silently
  drops the attempt). This is the layer future agent tools should call,
  not raw SQLAlchemy queries, so the same logic is reusable from a CLI or
  API later too.
- **`seed.py`**: populates sample employee leave balances for local
  development/testing, ahead of any real HR system integration.

### `src/askhr/eval/`
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
  eventual CI gate (not wired up yet). Benefits automatically from
  `RAGPipeline`'s chain caching when scoring many questions in a loop,
  since it's the same `RAGPipeline` class.

### `src/askhr/gui/main.py`
The app's current entry point. Depends on `PolicyQAService` (via
`get_policy_qa_service()`, additionally wrapped in `st.cache_resource` so
it survives Streamlit's rerun-the-whole-script-on-every-interaction
model), not on `RAGPipeline` directly. One button ("Index Policy
Documents"), one text input, citations rendered as expandable sections
showing doc + page + backing text.

## Key design decisions

- **Single `askhr` package namespace under `src/`** (`where = ["src"]`
  in `pyproject.toml`): `core`, `ingestion`, `processing`, `search`,
  `service`, `db`, `eval`, and `gui` are all subpackages of `askhr`, not
  flat top-level packages. This replaced an earlier flat layout where
  each was its own separate top-level importable package — a namespace
  collision risk once more packages (agents, api, etc.) get added
  alongside them.

- **`uv` for dependency management**: a lockfile (`uv.lock`) pins exact
  resolved versions for reproducibility, replacing a bare
  `pip install -e .` against loosely-pinned (`>=`-only) dependencies.

- **Lazy initialization + caching over rebuilding per call**:
  `RAGPipeline` builds the LLM client and the full retrieval chain on
  first access, caches both on the instance, and only invalidates the
  chain cache when `ingest_documents()` actually changes the corpus —
  not on a timer, not on every query. The same principle extends to
  BM25's index specifically (see next point) and to the service layer's
  process-wide singleton (`get_policy_qa_service()`, via
  `functools.lru_cache`).

- **BM25 index built once at ingestion time, persisted to disk, not
  rebuilt on every cold start**: previously, `build_hybrid_retriever()`
  called `vsm.get_all_documents()` and rebuilt BM25 from scratch on the
  first query after every process start (or after the chain cache was
  otherwise empty) — a real, corpus-size-scaling cost paid on the
  request path. Now, `ingest_documents()` builds BM25 directly from the
  chunk list it already has in hand (no Chroma round-trip) and pickles
  it to `settings.bm25_index_path`; `build_hybrid_retriever()` loads that
  pickle first, falling back to the old rebuild-from-Chroma behavior only
  if the pickle is missing or unreadable. This does not eliminate cold
  start entirely (the very first load after a restart still reads and
  deserializes the pickle) and does not address multi-replica RAM
  duplication or BM25's linear per-query scan cost at very large corpus
  sizes — see `HYBRID_SEARCH.md` for the full tradeoff discussion and
  what the next step (a disk-backed index like Postgres full-text search
  or Elasticsearch) would additionally solve.

- **A `service/` layer between the GUI and `RAGPipeline`**: added so
  that "how the pipeline gets built and cached" isn't owned by whichever
  consumer happened to be built first (the Streamlit GUI). Future
  LangGraph agent tools and the GUI both depend on `PolicyQAService`,
  keeping `RAGPipeline` free to change internally without every caller
  needing to change too.

- **`db/` built ahead of the agents that will use it**: the leave-balance
  SQLAlchemy models, session management, and repository exist
  independently of any agent, verified via an in-memory-SQLite smoke
  test, so the eventual LangGraph "fetch leave balance"/"apply leave"
  tools can be thin wrappers around already-correct, already-tested
  logic rather than being built and debugged at the same time as the
  agent orchestration itself.

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
