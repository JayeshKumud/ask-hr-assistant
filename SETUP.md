# Setup

## 1. Prerequisites

- Python **3.11.11**
- A free [Groq API key](https://console.groq.com/keys)

This project uses a plain `venv` + `pyproject.toml` — no conda, no
`requirements.txt`/`environment.yaml`. `pyproject.toml`'s `dependencies`
list is the single source of truth for versions.

## 2. Create the virtual environment & install

```powershell
uv sync
uv run streamlit run src/gui/main.py
```

Installs the project itself in editable mode plus every dependency in
`pyproject.toml`. Editable mode matters here: it's what makes `core`,
`ingestion`, `processing`, `search`, `eval`, and `gui` importable as
top-level packages regardless of which directory you run a script from
(`pyproject.toml` uses `where = ["src"]`, so those live at the top level,
not under a `src.` prefix).

## 3. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your-key-here
```

Every other setting (chunk size, embedding model, retrieval weights,
re-ranker model, token limits — see `src/core/config.py` for the full
list) has a sensible default; only override what you actually need to
change.

## 4. Add your policy documents

Drop PDF files into `resources/policies/`. Only `.pdf` is currently
supported (see `ARCHITECTURE.md` for why HTML support was dropped).
Filenames don't matter — every PDF in that folder gets ingested.

## 5. Run it

```powershell
streamlit run src/gui/main.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).
Click **Index Policy Documents** in the sidebar first — this ingests,
chunks, and embeds every PDF in `resources/policies/` into the local
Chroma vector store. Then ask a question in the text box.

## 7. Run the offline evaluation

```powershell
python -m eval.evaluate_faithfulness
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

## 8. Run individual module tests

Most modules under `src/` have a `if __name__ == "__main__":` block for
quick manual verification without running the whole app. From `src/`,
with the venv active:

```powershell
python -m ingestion.document_loader
python -m ingestion.text_splitter
python -m search.citations
python -m search.hybrid_retriever      # NOTE: temporarily resets your real vector store — re-ingest afterward
python -m search.reranker              # offline, uses a fake cross-encoder
python -m search.citation_enforcer     # offline, uses a fake LLM
python -m core.prompt_registry
python -m eval.dataset_loader
python -m eval.groq_deepeval_llm       # offline, uses a fake LLM
python -m core.pipeline                # full real run — needs GROQ_API_KEY
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
live in `langchain_community`, which hasn't been migrated the same way.
If an import breaks after a `langchain` upgrade, check which of these
three packages the symbol actually lives in now before assuming it's gone.

**`ModuleNotFoundError: No module named 'core'` (or `ingestion`, `search`, `gui`, `eval`)**
The project package itself isn't installed. Run `pip install -e .` from
the project root.

**Running a script directly fails with a path/import error**
Running a nested script directly (`python src/gui/main.py` instead of
`streamlit run src/gui/main.py`, or a file-based "Run" button in an IDE)
puts that script's own folder on `sys.path`/as `cwd`, not the project
root. `core/config.py` anchors its file paths (`policies_dir`,
`vector_store_dir`) to the project root via `__file__` specifically to
avoid this — but imports like `from core.pipeline import RAGPipeline`
still need the editable install (`pip install -e .`) to resolve
regardless of cwd.

**`groq.NotFoundError: model ... does not exist`**
Groq periodically deprecates models. Check `src/core/config.py`'s
`llm_groq_qwen_model` default (or your `.env`'s `LLM_MODEL`) against Groq's
[current model list](https://console.groq.com/docs/models).

**Answer comes back empty, but citations are populated**
This means the LLM ran out of its `max_tokens` generation budget before
finishing — likely mid-internal-reasoning, if using a reasoning model
like `qwen/qwen3.6-27b`. `reasoning_format="hidden"` only hides the
reasoning text, it doesn't stop the model from *spending tokens* on it.
The current fix in `core/pipeline.py` uses `reasoning_effort="none"`,
which disables reasoning entirely rather than just hiding it. If you
switch to a different reasoning model and this recurs, check both
`llm_max_tokens` and whether that model has an equivalent
reasoning-disable flag.

**`ragas` fails to import** (`ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`)
This is a known, currently-unresolved upstream bug in `ragas` (all
versions tested fail the same way, or require an incompatible
`langchain-core` downgrade to avoid it) — not something fixable via a
pyproject pin. This project uses **DeepEval** instead
(`src/eval/evaluate_faithfulness.py`), which was verified to work
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
