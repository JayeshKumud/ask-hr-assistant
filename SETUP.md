# Setup

## 1. Prerequisites

- Python **3.11–3.12**
- [`uv`](https://docs.astral.sh/uv/) — install via:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  (macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A free [Groq API key](https://console.groq.com/keys)

This project uses `uv` + `pyproject.toml` — no conda, no
`requirements.txt`/`environment.yaml`, no manually-managed venv.
`pyproject.toml`'s `dependencies` list is the single source of truth for
versions, and `uv.lock` pins exact resolved versions for reproducibility.

## 2. Pin the Python version and sync dependencies

```powershell
uv python pin 3.12
uv sync
```

`uv sync` creates `.venv` automatically, installs the project in
editable mode, and installs every dependency — no separate
`pip install -e .` step needed. This is what makes `askhr` (and its
subpackages: `core`, `ingestion`, `processing`, `search`, `service`,
`db`, `eval`, `gui`) importable regardless of which directory you run a
script from (`pyproject.toml` uses `where = ["src"]`, so `askhr` lives at
the top level of the installed package set).

If you change a dependency later, `uv add <package>` (or edit
`pyproject.toml` and re-run `uv sync`) updates both the environment and
`uv.lock`.

Run any script or command with `uv run ...` — this uses the project's
`.venv` automatically, no manual activation needed.

## 3. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your-key-here
```

Every other setting (chunk size, embedding model, retrieval weights,
re-ranker model, token limits, BM25 index path, database URL — see
`src/askhr/core/config.py` for the full list) has a sensible default;
only override what you actually need to change.

## 4. Add your policy documents

Drop PDF files into `resources/policies/`. Only `.pdf` is currently
supported (see `ARCHITECTURE.md` for why HTML support was dropped).
Filenames don't matter — every PDF in that folder gets ingested.

## 5. Run it

```powershell
uv run streamlit run src/askhr/gui/main.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Click **Index Policy Documents** in the sidebar first — this ingests,
chunks, and embeds every PDF in `resources/policies/` into the local
Chroma vector store, and also builds + persists the BM25 keyword index
to `resources/vectorstore/bm25_index.pkl` (see `HYBRID_SEARCH.md` for
why). Then ask a question in the text box.

## 6. Run the offline evaluation

```powershell
uv run python -m askhr.eval.evaluate_faithfulness
```

Runs all 20 golden questions (`resources/sample_questions/positive_rag_questions.csv`)
through the real pipeline, scores each answer's faithfulness to the
retrieved excerpts via DeepEval, and runs the 5 out-of-scope questions
(`negative_out_of_document_questions.csv`) to confirm the system
correctly refuses rather than hallucinating. Prints a pass/fail report
and exits non-zero on failure — this is what a CI pipeline will
eventually gate on (not wired up yet).

Expect this to take a while and use real Groq API quota — it makes two
LLM calls per question (answer generation + citation-enforcement
verification), for 25 questions.

## 7. (Optional, for Phase 4/5 work) Set up the leave-balance database

By default, `DATABASE_URL` points at a local SQLite file
(`resources/db/askhr.db`) — no separate setup needed. To seed it with
sample employee leave balances (used when building/testing agent
tools):

```powershell
uv run python -m askhr.db.seed
```

To point at a real Postgres instance instead (e.g. a free
[Neon](https://neon.tech) database — see note below on why Neon over
similar free tiers), set in `.env`:

```
DATABASE_URL=postgresql://<user>:<password>@<host>/<dbname>
```

No code changes needed — `src/askhr/db/session.py` only ever reads
`settings.database_url`. Neon's free tier doesn't pause on inactivity
(unlike some alternatives), which matters here since a paused database
would reintroduce a cold-start-style delay on the first query after a
quiet period — the exact class of problem this project's caching work
elsewhere was written to avoid.

## 8. Run individual module tests

Most modules under `src/askhr/` have an `if __name__ == "__main__":`
block for quick manual verification without running the whole app:

```powershell
uv run python -m askhr.ingestion.document_loader
uv run python -m askhr.ingestion.text_splitter
uv run python -m askhr.search.citations
uv run python -m askhr.search.hybrid_retriever      # NOTE: temporarily resets your real vector store — re-ingest afterward
uv run python -m askhr.search.reranker              # offline, uses a fake cross-encoder
uv run python -m askhr.search.citation_enforcer     # offline, uses a fake LLM
uv run python -m askhr.core.prompt_registry
uv run python -m askhr.eval.dataset_loader
uv run python -m askhr.eval.groq_deepeval_llm       # offline, uses a fake LLM
uv run python -m askhr.core.pipeline                # full real run — needs GROQ_API_KEY
uv run python -m askhr.db.repositories.leave_repository  # offline, in-memory SQLite
uv run python -m askhr.db.seed                            # seeds sample leave balances
```

## Troubleshooting

**`ModuleNotFoundError` for a `langchain.*` import**
`langchain>=1.0` split several things out of the core package:
`langchain.text_splitter` → `langchain_text_splitters`; the old `Chain`
classes (`RetrievalQAWithSourcesChain`, `load_qa_with_sources_chain`,
`EnsembleRetriever`, `ContextualCompressionRetriever`,
`CrossEncoderReranker`, `stuff_prompt`) → `langchain_classic`;
`PromptTemplate` → `langchain_core.prompts` (also re-exported from
`langchain_classic.prompts`). `BM25Retriever` and `HuggingFaceCrossEncoder`
live in `langchain_community`, which hasn't been migrated the same way
(and is currently marked for eventual sunsetting — expect a
`DeprecationWarning` on import; not something to fix urgently).
If an import breaks after a `langchain` upgrade, check which of these
three packages the symbol actually lives in now before assuming it's gone.

**`ModuleNotFoundError: No module named 'askhr'`**
The project package itself isn't installed. Run `uv sync` from the
project root.

**Running a script directly fails with a path/import error**
Running a nested script directly (`python src/askhr/gui/main.py` instead
of `uv run streamlit run src/askhr/gui/main.py`, or a file-based "Run"
button in an IDE) puts that script's own folder on `sys.path`/as `cwd`,
not the project root. `core/config.py` anchors its file paths
(`policies_dir`, `vector_store_dir`, `bm25_index_path`, SQLite's
`database_url`) to the project root via `__file__` specifically to avoid
this — but imports like `from askhr.core.pipeline import RAGPipeline`
still need the editable install (`uv sync`) to resolve regardless of cwd.

**`groq.NotFoundError: model ... does not exist`**
Groq periodically deprecates models. Check `src/askhr/core/config.py`'s
`llm_groq_qwen_model` default (or your `.env`'s `LLM_MODEL`) against Groq's
[current model list](https://console.groq.com/docs/models).

**`RateLimitError` from Groq mid-query**
Expected under real usage on the free tier — Groq enforces per-minute
token limits. `core/llm_wrappers.py`'s fallback chain (Groq/Qwen ->
HuggingFace Gemma -> HuggingFace Mistral) should transparently retry
with a fallback model; you'll see a `— falling back if another model is
configured` log line, and the query should still complete. If it
doesn't, check that a fallback model's own API key is also set in `.env`.

**Answer comes back empty, but citations are populated**
This means the LLM ran out of its `max_tokens` generation budget before
finishing — likely mid-internal-reasoning, if using a reasoning model
like `qwen/qwen3.6-27b`. `reasoning_format="hidden"` only hides the
reasoning text, it doesn't stop the model from *spending tokens* on it.
The current fix in `core/pipeline.py`/`llm_wrappers.py` uses
`reasoning_effort="none"`, which disables reasoning entirely rather than
just hiding it. If you switch to a different reasoning model and this
recurs, check both `llm_max_tokens` and whether that model has an
equivalent reasoning-disable flag.

**Citation verification crashes with `IndexError: list index out of range`**
Fixed — `search/citation_enforcer.py`'s `_is_supported()` now guards
against the verification LLM call returning an empty response (observed
occasionally with `reasoning_format="hidden"` on short verification-style
prompts) and treats it as "unsupported" rather than crashing. If you see
frequent `"Empty verification response..."` warnings in the logs, that
model's reasoning-suppression setting is worth revisiting for that
specific call.

**`ragas` fails to import** (`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`)
This is a known, currently-unresolved upstream bug in `ragas` (all
versions tested fail the same way, or require an incompatible
`langchain-core` downgrade to avoid it) — not something fixable via a
pyproject pin. This project uses **DeepEval** instead
(`src/askhr/eval/evaluate_faithfulness.py`), which was verified to work
cleanly with this project's dependency versions.

**`UserWarning: Using fallback GPT-2 tokenizer for token counting`**
Harmless — `ChatGroq` doesn't expose a model-specific tokenizer.

**Streamlit shows a blank white page**
Server started fine but the browser never rendered anything. Hard
refresh (Ctrl+Shift+R), check the browser console (F12) for WebSocket
errors, and rule out VPN/antivirus software inspecting localhost traffic.

**Torchvision-related traceback spam in the terminal on `streamlit run`**
Cosmetic — Streamlit's file watcher tries to introspect every
`transformers` submodule, some of which optionally import `torchvision`
(not installed, not needed here). Doesn't affect functionality; safe to
ignore.

**BM25 results seem stale after re-ingesting**
Shouldn't happen — `ingest_documents()` rebuilds and overwrites
`resources/vectorstore/bm25_index.pkl` on every re-index, and also
resets the cached retrieval chain (`RAGPipeline._qa_chain`) so the next
query is guaranteed to load the fresh pickle. If you genuinely see stale
results, check whether you're running multiple Streamlit processes
against the same vector store (each process caches its own chain
independently) or whether a pickle version mismatch caused a silent
fallback — check the logs for `"Failed to load persisted BM25 index"`.
