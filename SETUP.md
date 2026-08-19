# Setup

## 1. Prerequisites

- Python **3.11.11**
- A free [Groq API key](https://console.groq.com/keys)

This project uses a plain `venv` + `pyproject.toml` — no conda, no
`requirements.txt`/`environment.yaml`.

## 2. Create the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

(macOS/Linux: `source .venv/bin/activate`)

## 3. Install the project

```powershell
pip install -e .
```

This installs the project itself in editable mode plus every dependency
listed in `pyproject.toml`. Editable mode matters here: it's what makes
`core`, `ingestion`, `processing`, `search`, and `gui` importable as
top-level packages regardless of which directory you run a script from.

If you change a dependency in `pyproject.toml` later, just re-run
`pip install -e .` to pick it up.

## 4. Configure environment variables

Copy the example env file and fill in your Groq key:

```powershell
copy .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your-key-here
```

The rest of `.env.example`'s variables (chunk size, embedding model, vector
store path, etc.) have sensible defaults from `src/core/config.py` — you only
need to override them if you want non-default behavior.

## 5. Run it

```powershell
streamlit run src/gui/main.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).
See the README's [Try it](README.md#try-it) section for two sample URLs to
paste in and a sample question to ask.

## 6. Run the tests

```powershell
pytest
```

## Troubleshooting

**`ModuleNotFoundError` for a `langchain.*` import**
`langchain>=1.0` split several things out of the core package:
`langchain.text_splitter` → `langchain_text_splitters`; the old `Chain`
classes (`RetrievalQAWithSourcesChain`, `load_qa_with_sources_chain`,
`stuff_prompt`) → `langchain_classic`; `PromptTemplate` →
`langchain_core.prompts`. Make sure `langchain-text-splitters` and
`langchain-classic` are in `pyproject.toml`'s dependencies, then
`pip install -e .` again.

**`ModuleNotFoundError: No module named 'core'` (or `ingestion`, `search`, `gui`)**
The project package itself isn't installed. Run `pip install -e .` from the
project root.

**Running a script directly fails with an import error**
Running a nested script directly (e.g. `python src/gui/main.py` instead of
`streamlit run src/gui/main.py`) puts *that script's own folder* on
`sys.path`, not the project root — this can break absolute imports like
`from core.pipeline import RAGPipeline` if the project isn't installed as an
editable package. Fix: confirm `pip install -e .` succeeded (check for a
`rag_research_iq.egg-info` folder under `src/` — its presence means the
editable install is active).

**`groq.NotFoundError: model ... does not exist`**
Groq periodically deprecates models. Check `src/core/config.py`'s
`llm_model` default (or your `.env`'s `LLM_MODEL`) against Groq's
[current model list](https://console.groq.com/docs/models) and update it.

**`UserWarning: Using fallback GPT-2 tokenizer for token counting`**
Harmless — `ChatGroq` doesn't expose a model-specific tokenizer, so LangChain
falls back to GPT-2's for token-based chunk trimming. Doesn't affect answer
quality at this project's scale.

**Streamlit shows a blank white page**
The server started fine but the browser never rendered anything. Hard
refresh (Ctrl+Shift+R), check the browser console (F12) for WebSocket
errors, and rule out VPN/antivirus software that inspects localhost traffic.

**Windows: `UnstructuredURLLoader` fails on import or at load time**
Make sure `python-magic-bin` installed (it's marked Windows-only in
dependencies). It provides `libmagic`, which `unstructured` needs and isn't
available on Windows without it.
