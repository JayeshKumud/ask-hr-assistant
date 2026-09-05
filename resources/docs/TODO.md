# TODO

## Done (this round of work)
- [x] update all documents — README, SETUP, ARCHITECTURE, HYBRID_SEARCH,
      WORKFLOW, RE_RANKER, PROCESS all updated to reflect: uv migration,
      the src/askhr/ namespace restructure, the service/ and db/ layers,
      retrieval-chain caching, and BM25 index persistence
- [x] migrate to uv package manager
- [x] restructure src/ into a single askhr package namespace, ahead of
      adding agents/db/api packages
- [x] add a service/ layer (PolicyQAService) so future agents and the
      GUI share one entry point instead of each owning RAGPipeline
- [x] add a db/ layer (SQLAlchemy: LeaveBalance, LeaveRequest,
      LeaveRepository) for the leave-balance agent tools — built and
      unit-verified, not yet wired to an agent
- [x] fix: retrieval chain was rebuilt from scratch on every query —
      now cached on RAGPipeline, invalidated only on re-ingest
- [x] fix: BM25 index was rebuilt from Chroma on every cold start — now
      built once at ingestion time and persisted to disk
- [x] fix: citation_enforcer crashed (IndexError) on an empty
      verification LLM response — now treated as "unsupported"
- [x] correct project/directory name (rag-research-iq -> ask-hr-assistant)

## Still open
- [ ] build the actual LangGraph agent graph (apply leave, fetch leave
      balance) — db/ and service/ are in place for this, agent itself
      not started
- [ ] add unit tests (pytest) — currently only manual
      `if __name__ == "__main__":` smoke tests per module
- [ ] CI pipeline wiring — gate on eval/evaluate_faithfulness.py results
- [ ] centralized structured logging (currently per-module, no unified
      setup)
- [ ] MCP server
- [ ] check EKS/Kubernetes video and implement (see PROCESS.md for the
      ingestion/serving separation this would build on)
- [ ] add tech stack doc/badge
- [ ] Gemini as an additional fallback model — llm_wrappers.py's chain
      is Groq/Qwen -> HF Gemma -> HF Mistral currently; add a Gemini
      rung using the same LiteLLM-based fallback mechanism
- [ ] revisit whether the current fallback naming/class structure in
      llm_wrappers.py still reads as industry-standard as more models
      get added, and refactor if it starts feeling ad hoc

## Longer-term (only worth doing once corpus/traffic actually justify it)
- [ ] if BM25/RAM or per-query latency become real bottlenecks at much
      larger corpus size: migrate keyword search off in-process
      rank_bm25 to Postgres full-text search or a dedicated search
      engine (Elasticsearch/OpenSearch) — see HYBRID_SEARCH.md
- [ ] if running multiple serving replicas: separate ingestion into its
      own process/job per PROCESS.md, since in-process caching (both the
      qa_chain cache and the BM25 pickle) doesn't share across replicas
- [ ] repo-per-integration split (deepeval, litellm, etc.) if this ever
      grows into multiple consumable libraries rather than one app
