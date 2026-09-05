# NexaCore Company Policy Assistant

A Retrieval-Augmented Generation (RAG) system that answers employee
questions about company leave and visa policy — grounded strictly in
your actual policy documents, with page-level citations, and an explicit
refusal when the documents don't support a confident answer rather than
guessing.

Originally built as a URL-scraping real-estate research tool
(`rag-research-iq`); the retrieval/generation architecture was kept, the
domain and ingestion source were replaced.

## What it does

- **Ingests PDF policy documents** (leave policy, visa/immigration
  policy, etc.) from a local folder — no web scraping, no external URLs.
- **Answers questions in natural language**, grounded only in what's
  actually in those documents.
- **Hybrid retrieval**: combines BM25 keyword search with vector
  (semantic) search, so both exact terms ("Form I-129") and paraphrased
  questions ("what if I'm sick?") are handled well.
- **Cross-encoder re-ranking**: a second, more precise pass narrows a
  wide retrieval candidate pool down to the best few chunks before they
  reach the LLM.
- **Page-level citations**: every answer shows exactly which document
  and page it came from, plus the actual backing text — not just a
  repeated filename.
- **Citation enforcement**: a second LLM call checks whether the
  generated answer is actually supported by the retrieved excerpts, and
  explicitly refuses ("I don't have enough information...") rather than
  returning an unsupported or hallucinated answer.
- **Versioned prompts**: all prompt text lives in a single YAML config
  file, not scattered across Python source.
- **Offline faithfulness evaluation**: a curated 20-question golden set
  (plus 5 out-of-scope questions) scored automatically via DeepEval,
  checking both answer faithfulness and correct-refusal behavior.
- **Cached retrieval chain + persisted BM25 index**: the retrieval/
  generation chain is built once per process and reused across queries,
  and the BM25 keyword index is built once during ingestion and
  persisted to disk — not rebuilt from scratch on every question or
  every app restart. See [ARCHITECTURE.md](resources/docs/ARCHITECTURE.md)
  and [HYBRID_SEARCH.md](resources/docs/HYBRID_SEARCH.md) for details.

## Quick start

See **[SETUP.md](SETUP.md)** for full instructions. Short version:

```bash
uv sync
copy .env.example .env        # then add your GROQ_API_KEY (Windows)
# cp .env.example .env        # macOS/Linux

uv run streamlit run src/askhr/gui/main.py
```

In the app: click **Index Policy Documents** in the sidebar first, then
ask a question in the text box.

## Documentation

| Doc | Covers |
|---|---|
| [SETUP.md](SETUP.md) | Environment setup, API keys, running the app, running the evaluation script, troubleshooting |
| [ARCHITECTURE.md](resources/docs/ARCHITECTURE.md) | Every module's purpose and responsibilities, key design decisions |
| [WORKFLOW.md](resources/docs/WORKFLOW.md) | Step-by-step execution flow for ingestion and for answering a query |
| [HYBRID_SEARCH.md](resources/docs/HYBRID_SEARCH.md) | How BM25 + vector hybrid retrieval works, why it's needed |
| [RE_RANKER.md](resources/docs/RE_RANKER.md) | How cross-encoder re-ranking works, why hybrid retrieval's ranking alone isn't enough |
| [PROCESS.md](resources/docs/PROCESS.md) | Why ingestion and serving should be separate processes in production |

## Project layout

```
ask-hr-assistant/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env / .env.example
├── config/
│   └── prompts.yaml            # versioned prompt templates (source of truth)
├── src/
│   └── askhr/                  # single top-level package namespace
│       ├── core/
│       │   ├── config.py           # central settings (env-driven, project-root-anchored)
│       │   ├── pipeline.py         # RAGPipeline — orchestrates the whole flow
│       │   ├── llm_wrappers.py     # LLM fallback chain (Groq -> HF Gemma -> HF Mistral)
│       │   ├── configure_logging.py
│       │   └── prompt_registry.py  # loads config/prompts.yaml
│       ├── ingestion/
│       │   ├── document_loader.py  # loads PDFs from resources/policies/
│       │   ├── text_splitter.py    # chunks documents, tracks chunk position
│       │   └── url_loader.py
│       ├── processing/
│       │   ├── embeddings.py       # embedding function factory
│       │   └── vector_store.py     # Chroma vector store management
│       ├── search/
│       │   ├── prompts.py          # builds LangChain PromptTemplates from the YAML config
│       │   ├── citations.py        # Citation dataclass + extraction
│       │   ├── hybrid_retriever.py # BM25 + vector, combined via RRF; BM25 index persistence
│       │   ├── reranker.py         # cross-encoder re-ranking on top of hybrid retrieval
│       │   ├── citation_enforcer.py# refuses unsupported answers
│       │   └── qa_chain.py         # ties retrieval + generation + enforcement together
│       ├── service/
│       │   └── policy_qa_service.py# PolicyQAService — the stable entry point for
│       │                             # "ask a policy question" / "reindex", shared by
│       │                             # the GUI today and future LangGraph agents later
│       ├── db/
│       │   ├── models.py           # LeaveBalance, LeaveRequest (SQLAlchemy)
│       │   ├── session.py          # engine/session management, init_db()
│       │   ├── seed.py             # sample leave-balance data for local dev
│       │   └── repositories/
│       │       └── leave_repository.py # get_balance() / apply_leave(), for future agents
│       ├── eval/
│       │   ├── dataset_loader.py       # loads the golden question CSVs
│       │   ├── groq_deepeval_llm.py    # adapts ChatGroq for DeepEval
│       │   └── evaluate_faithfulness.py# offline faithfulness + refusal evaluation script
│       └── gui/
│           └── main.py             # Streamlit app — the entry point
├── resources/
│   ├── policies/                # source PDF policy documents
│   ├── sample_questions/        # golden eval question sets (CSV)
│   ├── db/                      # local SQLite file (not committed)
│   └── vectorstore/              # persisted Chroma DB + BM25 pickle (not committed)
└── tests/                       # (in progress)
```

## Status

Built incrementally; current state:

- ✅ Migrated to `uv` for dependency management (lockfile, no more bare `pip install -e .`)
- ✅ Restructured into a single `askhr` package namespace under `src/`
- ✅ `PolicyQAService` — interface-agnostic entry point shared by the GUI, ready to be shared with future agents
- ✅ `db/` layer (SQLAlchemy) for leave balances/requests — built and unit-verified; not yet wired to an agent
- ✅ PDF ingestion, chunking, vector storage
- ✅ Page-level citations
- ✅ Hybrid (BM25 + vector) retrieval
- ✅ Cross-encoder re-ranking
- ✅ Citation enforcement (refuses unsupported answers)
- ✅ Versioned prompts (`config/prompts.yaml`)
- ✅ Offline faithfulness evaluation script (DeepEval)
- ✅ Retrieval chain cached per-process instead of rebuilt on every query
- ✅ BM25 index built once at ingestion time and persisted to disk, instead of rebuilt from Chroma on every cold start
- ⏳ LangGraph agents (apply leave, fetch leave balance) — `db/` and `service/` layers are in place for this; agent graph itself not yet built
- ⏳ CI pipeline wiring (fail build on quality regression) — not yet done
- ⏳ Centralized structured logging — partial (individual modules log; no unified setup yet)
- ⏳ Unit test suite — not yet built
