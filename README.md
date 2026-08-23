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

## Quick start

See **[SETUP.md](SETUP.md)** for full instructions. Short version:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e .
copy .env.example .env        # then add your GROQ_API_KEY

streamlit run src/gui/main.py
```

In the app: click **Index Policy Documents** in the sidebar first, then
ask a question in the text box.

## Documentation

| Doc | Covers |
|---|---|
| [SETUP.md](SETUP.md) | Environment setup, API keys, running the app, running the evaluation script, troubleshooting |
| [ARCHITECTURE.md](resources/docs/ARCHITECTURE.md) | Every module's purpose and responsibilities, key design decisions |
| [WORKFLOW.md](resources/docs/WORKFLOW.md) | Step-by-step execution flow for ingestion and for answering a query |
| [HYBRID_SEARCH.md](HYBRID_SEARCH.md) | How BM25 + vector hybrid retrieval works, why it's needed |
| [RE_RANKER.md](RE_RANKER.md) | How cross-encoder re-ranking works, why hybrid retrieval's ranking alone isn't enough |

## Project layout

```
rag-research-iq/
├── pyproject.toml
├── .env / .env.example
├── config/
│   └── prompts.yaml            # versioned prompt templates (source of truth)
├── src/
│   ├── core/
│   │   ├── config.py           # central settings (env-driven, project-root-anchored)
│   │   ├── pipeline.py         # RAGPipeline — orchestrates the whole flow
│   │   └── prompt_registry.py  # loads config/prompts.yaml
│   ├── ingestion/
│   │   ├── document_loader.py  # loads PDFs from resources/policies/
│   │   └── text_splitter.py    # chunks documents, tracks chunk position
│   ├── processing/
│   │   ├── embeddings.py       # embedding function factory
│   │   └── vector_store.py     # Chroma vector store management
│   ├── search/
│   │   ├── prompts.py          # builds LangChain PromptTemplates from the YAML config
│   │   ├── citations.py        # Citation dataclass + extraction
│   │   ├── hybrid_retriever.py # BM25 + vector, combined via RRF
│   │   ├── reranker.py         # cross-encoder re-ranking on top of hybrid retrieval
│   │   ├── citation_enforcer.py# refuses unsupported answers
│   │   └── qa_chain.py         # ties retrieval + generation + enforcement together
│   ├── eval/
│   │   ├── dataset_loader.py       # loads the golden question CSVs
│   │   ├── groq_deepeval_llm.py    # adapts ChatGroq for DeepEval
│   │   └── evaluate_faithfulness.py# offline faithfulness + refusal evaluation script
│   └── gui/
│       └── main.py             # Streamlit app — the entry point
├── resources/
│   ├── policies/                # source PDF policy documents
│   ├── sample_questions/        # golden eval question sets (CSV)
│   └── vectorstore/              # persisted Chroma DB (not committed)
└── tests/                       # (in progress)
```

## Status

Built incrementally; current state:

- ✅ PDF ingestion, chunking, vector storage
- ✅ Page-level citations
- ✅ Hybrid (BM25 + vector) retrieval
- ✅ Cross-encoder re-ranking
- ✅ Citation enforcement (refuses unsupported answers)
- ✅ Versioned prompts (`config/prompts.yaml`)
- ✅ Offline faithfulness evaluation script (DeepEval)
- ⏳ CI pipeline wiring (fail build on quality regression) — not yet done
- ⏳ Centralized structured logging — partial (individual modules log; no unified setup yet)
- ⏳ Unit test suite — not yet built
