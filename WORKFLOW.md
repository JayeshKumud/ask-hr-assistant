# Workflow

This describes the two operations the app supports, step by step.

## 1. Ingestion -- `pipeline.process_urls(urls)`

Triggered by: clicking **Process URLs** in the Streamlit sidebar.

```
  urls: List[str]
      |
      v
  1. Initialize components (lazy -- only builds the LLM client and vector
     store connection the first time this pipeline instance is used)
      |
      v
  2. Reset the vector store's collection
     (clears out anything from a previous run -- this is a full replace,
     not an append)
      |
      v
  3. Load raw documents from each URL
     (ingestion.url_loader -- scrapes and parses article text)
      |
      v
  4. Split documents into chunks
     (ingestion.text_splitter -- sized by core.config.chunk_size /
     chunk_overlap)
      |
      v
  5. Embed each chunk and add it to the vector store
     (processing.embeddings + processing.vector_store -- HuggingFace
     embedding model, stored in Chroma, persisted to disk under
     resources/vectorstore/)
      |
      v
  Done -- yields a status string after each step (drives the GUI's live
  status placeholder)
```

**Why it's a generator, not a single call**: the GUI wants to show progress
as ingestion happens (updating the Streamlit placeholder live), rather than
blocking silently until everything is done -- scraping + embedding a handful
of articles can take several seconds.

**Why the vector store is reset every time**: this is a "process this batch
of URLs into a fresh index" workflow, not an incremental one -- running
`process_urls` again replaces the previous set of documents rather than
adding to them.

## 2. Answering a query -- `pipeline.generate_answer(query)`

Triggered by: typing into the GUI's question box.

```
  query: str
      |
      v
  1. Guard: raise RuntimeError if the vector store hasn't been
     initialized/populated yet (i.e. process_urls was never called)
      |
      v
  2. Build the retrieval QA chain
     (search.qa_chain -- combines the LLM, the vector store's retriever,
     and search.prompts' templates into one chain)
      |
      v
  3. Retriever embeds the query and pulls the top-k most similar chunks
     from the vector store
      |
      v
  4. Chunks + query get formatted into the QA prompt and sent to the LLM
     (Groq-hosted model, per core.config.llm_model)
      |
      v
  5. Parse the result: separate the answer text from the list of source
     URLs (pulled from each retrieved chunk's metadata)
      |
      v
  Returns (answer: str, sources: List[str])
```

**Why `generate_answer` can fail with `RuntimeError`**: it's a deliberate
guard -- asking a question before any URLs have been processed means there's
nothing in the vector store to retrieve from. The GUI catches this
specifically and shows "You must process urls first" instead of crashing.

## How the GUI drives this

The GUI wraps one `RAGPipeline` instance in `st.cache_resource`. Streamlit
re-executes the *entire* script on every interaction (typing a character,
clicking a button) -- without that cache, ingestion would need to be redone
from scratch before every single question. The cache is what lets
"process once, ask many questions" behave the way you'd expect, and it's
also why the persisted Chroma files under `resources/vectorstore/` survive
between app restarts even though the in-memory pipeline instance doesn't.
