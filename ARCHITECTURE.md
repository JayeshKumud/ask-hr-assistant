# Architecture

## Overview

This project follows a layered pipeline architecture: each layer has one
job, depends only on the layer(s) below it, and is swappable independently.
`core.pipeline.RAGPipeline` is the only piece that knows about *all* the
layers — everything else stays narrowly scoped.

```
+-----------------------------------------------+
|                 src/gui/main.py                |
|              (Streamlit app -- the             |
|                only entry point)                |
+-----------------------+-------------------------+
                         | uses
                         v
+-----------------------------------------------+
|              core.pipeline.RAGPipeline          |
|  Orchestrates the full flow. The only module    |
|  that talks to ingestion, processing, AND       |
|  search.                                        |
+------+-------------------+---------------+------+
       |                   |               |
       v                   v               v
+-------------+   +------------------+  +------------------+
|  ingestion   |   |   processing     |  |     search        |
|              |   |                  |  |                   |
| - load URLs  |-->| - build embed-   |->| - retrieve top-k  |
| - split into |   |   ding function  |  |   chunks          |
|   chunks     |   | - manage the     |  | - build QA chain  |
|              |   |   Chroma vector  |  | - call the LLM    |
|              |   |   store          |  | - return answer   |
|              |   |                  |  |   + sources       |
+-------------+   +------------------+  +------------------+
                         ^
                         | constants
                +--------+--------+
                |  core/config.py  |
                | (env-driven      |
                |  settings)       |
                +------------------+
```

## Module responsibilities

### `src/core/config.py`
Single source of truth for every tunable constant: model names, chunk size,
vector store path, top-k, token limits. Reads from environment variables
(via `.env`) with defaults, so nothing else in the codebase should hardcode
a model name or path directly — they import `settings` from here instead.

### `src/ingestion/`
Everything about getting raw content *in*: `url_loader.py` loads article
text from URLs, `text_splitter.py` splits loaded documents into
embedding-sized chunks. A new source type (PDFs, local files, an API) would
get added as its own file here, without the rest of the pipeline changing.

### `src/processing/`
Owns the vector store lifecycle: `embeddings.py` builds the embedding
function, `vector_store.py` creates/resets the Chroma collection, adds
documents, and exposes a retriever. Nothing outside this package should
touch Chroma directly.

### `src/search/`
Turns a query into a sourced answer: `prompts.py` holds the QA prompt
templates, `qa_chain.py` builds the retrieval QA chain (prompt + LLM +
retriever), invokes it, and extracts the answer text and source URLs.

### `src/core/pipeline.py`
`RAGPipeline` — the orchestrator. Exposes exactly two operations:
- `process_urls(urls)` — ingest -> chunk -> embed -> store, yielding
  progress messages as it goes
- `generate_answer(query)` — retrieve -> prompt -> LLM -> sourced answer

It initializes the LLM client and vector store **lazily** (on first use,
cached on the instance) rather than via module-level globals and an
`initialize_components()` function. That means no import-order dependency,
and each `RAGPipeline()` instance is self-contained — which is what makes
`st.cache_resource` in the GUI layer work cleanly.

### `src/gui/main.py`
The app's only entry point. Streamlit front-end wrapping a single cached
`RAGPipeline` instance (`st.cache_resource`) so the vector store survives
across Streamlit's rerun-the-whole-script-on-every-interaction model — see
WORKFLOW.md for why that matters.

## Key design decisions

- **`src` layout, packages resolved from `src/`** (`where = ["src"]` in
  `pyproject.toml`): `core`, `ingestion`, `processing`, `search`, and `gui`
  are top-level importable packages once the project is installed
  (`pip install -e .`), not nested under a `src.` prefix.

- **Lazy initialization over module globals**: `RAGPipeline` builds the LLM
  client and vector store on first access and caches them on the instance,
  rather than mutating module-level `llm`/`vector_store` globals via
  `global`. No mutable global state to reason about.

- **venv + `pyproject.toml` only**: no conda, no `environment.yaml`, no
  `requirements.txt`. `pyproject.toml`'s `dependencies` list is the single
  place versions are declared; `pip install -e .` installs both the project
  and its dependencies in one step.
