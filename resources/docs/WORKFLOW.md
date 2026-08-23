# Workflow

Step-by-step execution flow for the two operations the app supports,
plus the offline evaluation flow.

## 1. Ingestion — `pipeline.ingest_documents()`

Triggered by: clicking **Index Policy Documents** in the Streamlit sidebar.

```
  (no input — reads from settings.policies_dir)
      |
      v
  1. Reset the vector store's collection
     (VectorStoreManager.reset() -- clears anything from a previous run;
     this is a full replace, not an incremental append)
      |
      v
  2. Load every PDF under resources/policies/
     (ingestion.document_loader.load_documents() -- LangChain's
     DirectoryLoader + PyPDFLoader; one Document per PAGE, with
     page/page_label/source metadata)
      |
      v
  3. Split into chunks
     (ingestion.text_splitter.split_documents() -- sized by
     settings.chunk_size/chunk_overlap; also assigns chunk_index per
     chunk, tracking its position within its source page)
      |
      v
  4. Embed each chunk and store it
     (processing.vector_store.VectorStoreManager.add_documents() --
     HuggingFace embedding model, stored in Chroma, persisted to disk
     under resources/vectorstore/, each chunk given a fresh UUID)
      |
      v
  Done -- yields a status string after each step, driving the GUI's
  live status placeholder
```

**Why the vector store is reset every time**: this is "process the
current set of policy PDFs into a fresh index", not an incremental
append — re-running ingestion replaces the previous contents rather than
adding to them.

**Why BM25 needs nothing done at ingestion time**: unlike Chroma, BM25
has no persisted index — it's rebuilt from whatever's in Chroma on
demand, at query time (see `HYBRID_SEARCH.md`). Nothing to update here.

## 2. Answering a query — `pipeline.generate_answer(query)`

Triggered by: typing into the GUI's question box.

```
  query: str
      |
      v
  1. Guard: raise RuntimeError if the vector store hasn't been
     initialized/populated yet (ingest_documents() was never called)
      |
      v
  2. Build the QA chain for this query
     (search.qa_chain.build_qa_chain() -- wires together the LLM, the
     re-ranking retriever, and the main prompt templates)
      |
      v
  3. RETRIEVAL: hybrid retrieval fetches a WIDE candidate pool
     (search.hybrid_retriever.build_hybrid_retriever(), k =
     settings.rerank_candidate_k -- runs BM25 keyword search AND vector
     semantic search on every query, unconditionally, then fuses the two
     ranked lists via weighted Reciprocal Rank Fusion. See
     HYBRID_SEARCH.md for the full mechanism.)
      |
      v
  4. RE-RANKING: cross-encoder narrows the pool down
     (search.reranker.build_reranking_retriever() -- scores each
     (query, chunk) pair jointly via a cross-encoder model, keeps only
     the top settings.top_k. See RE_RANKER.md for why this is more
     precise than the fused ranking from step 3 alone.)
      |
      v
  5. GENERATION: the narrowed chunks + query go to the LLM
     (formatted via EXAMPLE_PROMPT per chunk, combined into PROMPT's
     {summaries} slot, sent to the Groq-hosted model per
     settings.llm_model)
      |
      v
  6. Extract citations from what was actually retrieved
     (search.citations.extract_citations() -- built from the chunks'
     metadata, NOT the LLM's self-reported "SOURCES:" text, which isn't
     trustworthy; deduplicated by (source, page, chunk_index))
      |
      v
  7. CITATION ENFORCEMENT: verify the answer is actually supported
     (search.citation_enforcer.enforce_citations() -- a SECOND, separate
     LLM call: given the same excerpts and the generated answer, asks
     the LLM to judge SUPPORTED / NOT_SUPPORTED. If no citations were
     retrieved at all, refuses immediately with no verification call
     needed. If citations exist but aren't judged supportive, replaces
     the answer with a refusal -- but still returns the citations found,
     so the user can review them.)
      |
      v
  Returns (answer: str, citations: List[Citation])
```

**Why `generate_answer` can fail with `RuntimeError`**: asking a question
before any documents have been ingested means there's nothing to
retrieve from. The GUI catches this specifically and shows
"You must index the policy documents first" instead of crashing.

**Cost note**: every query makes at least 2 LLM calls (generation +
verification), sometimes effectively a 3rd (re-ranking's cross-encoder
model runs locally, not an LLM call, but adds its own latency). This is
a deliberate accuracy-over-cost tradeoff.

## 3. Offline evaluation — `python -m eval.evaluate_faithfulness`

Triggered manually (eventually: by CI, on each pull request — not wired
up yet).

```
  1. Ingest the real policy documents
     (same RAGPipeline.ingest_documents() as production use)
      |
      v
  2. For each of the 20 positive golden questions:
     (resources/sample_questions/positive_rag_questions.csv)
       a. Run it through the real pipeline -> (answer, citations)
       b. Build a DeepEval LLMTestCase(input=question,
          actual_output=answer, retrieval_context=[citation snippets])
       c. Score with DeepEval's FaithfulnessMetric (judged by
          GroqDeepEvalLLM -- an adapter wrapping the pipeline's own
          ChatGroq instance for DeepEval's judge-model interface)
      |
      v
  3. For each of the 5 negative (out-of-scope) questions:
     (resources/sample_questions/negative_out_of_document_questions.csv)
       a. Run it through the real pipeline
       b. Check whether the answer exactly matches one of
          citation_enforcer's refusal messages (i.e. did the system
          correctly decline, rather than hallucinate an answer from
          outside the documents?)
      |
      v
  4. Report: average faithfulness score, per-question failures below
     threshold, refusal rate, missed refusals. Exits non-zero if either
     average faithfulness or refusal rate falls below its configured
     threshold.
```

This is a different check than citation enforcement (step 7 above):
citation enforcement is a **live guardrail** running on every real query
in production; this evaluation script is an **offline, scored** check
against a fixed, curated test set, meant to catch regressions over time
as the system changes (prompt edits, model swaps, retrieval tuning) —
not something that runs for every user's question.

## How the GUI drives all of this

`src/gui/main.py` wraps one `RAGPipeline` instance in `st.cache_resource`.
Streamlit re-executes the *entire* script on every interaction — without
that cache, ingestion would need to be redone from scratch before every
single question. The cache is what lets "index once, ask many questions"
work as expected, and it's also why the persisted Chroma files under
`resources/vectorstore/` survive app restarts even though the in-memory
pipeline instance doesn't (BM25's index, having no persistence of its
own, gets rebuilt from Chroma automatically on the next query either way).
