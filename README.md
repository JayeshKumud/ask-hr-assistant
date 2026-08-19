# rag-research-iq

A Retrieval-Augmented Generation (RAG) research assistant that scrapes real-estate
news articles from URLs, indexes them in a local vector store, and answers
questions about their content with sourced citations — using Groq-hosted LLMs,
HuggingFace embeddings, and Chroma.

Runs as a Streamlit app.

## Features

- **Ingest** any set of article URLs into a local vector database
- **Ask questions** in natural language and get answers grounded in the
  ingested articles, with source URLs attached
- Runs on Groq's fast open-weight models (currently `openai/gpt-oss-120b`)
- No cloud vector DB required — Chroma persists to disk locally under
  `resources/vectorstore/`

## Quick start

See **[SETUP.md](SETUP.md)** for full setup instructions. The short version:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell/cmd)
# source .venv/bin/activate   # macOS/Linux

pip install -e .
# add your GROQ_API_KEY to a .env file (see SETUP.md)

streamlit run src/gui/main.py
```

## Try it

The project ships with two sample URLs you can use to test ingestion and
querying end-to-end:

- https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html
- https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html

Paste those into the app's sidebar URL fields, click **Process URLs**, then
ask something like:

> Tell me what was the 30 year fixed mortgage rate along with the date?

## Documentation

| Doc | Covers |
|---|---|
| [SETUP.md](SETUP.md) | venv setup, API keys, running the app, running tests, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Module layout, what each package is responsible for, key design decisions |
| [WORKFLOW.md](WORKFLOW.md) | Step-by-step data flow for ingestion and for answering a query |

## Project layout

```
rag-research-iq/
├── pyproject.toml
├── .env / .env.example
├── src/
│   ├── core/
│   │   ├── config.py         # central settings (env-driven)
│   │   └── pipeline.py       # RAGPipeline — orchestrates the whole flow
│   ├── ingestion/
│   │   ├── url_loader.py     # loads raw docs from URLs
│   │   └── text_splitter.py  # chunks documents
│   ├── processing/
│   │   ├── embeddings.py     # embedding function
│   │   └── vector_store.py   # Chroma vector store management
│   ├── search/
│   │   ├── prompts.py        # QA prompt templates
│   │   └── qa_chain.py       # retrieval QA chain + generate_answer
│   └── gui/
│       └── main.py            # Streamlit app — the entry point
├── resources/vectorstore/    # persisted Chroma DB (not committed)
└── tests/
```
