from typing import Tuple, List

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain

from core.config import settings
from processing.vector_store import VectorStoreManager
from search.citation_enforcer import enforce_citations
from search.citations import Citation, extract_citations
from search.prompts import PROMPT, EXAMPLE_PROMPT
from search.reranker import build_reranking_retriever


def build_qa_chain(llm, vsm: VectorStoreManager) -> RetrievalQAWithSourcesChain:
    """
    Builds the retrieval-augmented QA chain from an LLM and a
    VectorStoreManager.

    The retriever is the full Phase 3 + Phase 4 pipeline: hybrid
    (BM25 + vector) retrieval fetches a wide candidate pool, then a
    cross-encoder re-ranks and narrows it down to the most relevant
    settings.top_k chunks before they reach the LLM.
    """
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=build_reranking_retriever(vsm),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=settings.max_tokens_limit,
        return_source_documents=True,
    )

def generate_answer(
    chain: RetrievalQAWithSourcesChain, llm, query: str
) -> Tuple[str, List[Citation]]:
    """
    Runs a query through the QA chain, returns (answer, list of Citations).

    Citations are built from result["source_documents"] — the chunks the
    retriever actually returned — rather than parsing the LLM's own
    "SOURCES:" text output, which only ever repeated the filename with no
    page/location info and can't be trusted to be accurate.

    Before returning, the answer passes through citation enforcement
    (search/citation_enforcer.py): a second, focused LLM call checks
    whether the answer's claims are actually backed by the retrieved
    excerpts, and replaces it with an explicit refusal if not — rather
    than returning a plausible-sounding but unsupported answer.
    """
    result = chain.invoke({"question": query}, return_only_outputs=True)
    citations = extract_citations(result["source_documents"])
    answer, citations = enforce_citations(llm, query, result["answer"], citations)
    return answer, citations