from typing import Tuple, List

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain

from core.config import settings
from processing.vector_store import VectorStoreManager
from search.citations import Citation, extract_citations
from search.hybrid_retriever import build_hybrid_retriever
from search.prompts import PROMPT, EXAMPLE_PROMPT


def build_qa_chain(llm, vsm: VectorStoreManager) -> RetrievalQAWithSourcesChain:
    """
    Builds the retrieval-augmented QA chain from an LLM and a
    VectorStoreManager.

    Takes the manager (not the raw Chroma store) because the hybrid
    retriever needs get_all_documents() from it to build BM25's index,
    in addition to the vector search it already provided.
    """
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=build_hybrid_retriever(vsm),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=settings.max_tokens_limit,
        return_source_documents=True,
    )

def generate_answer(chain: RetrievalQAWithSourcesChain, query: str) -> Tuple[str, List[Citation]]:
    """
    Runs a query through the QA chain, returns (answer, list of Citations).

    Citations are built from result["source_documents"] — the chunks the
    retriever actually returned — rather than parsing the LLM's own
    "SOURCES:" text output, which only ever repeated the filename with no
    page/location info and can't be trusted to be accurate (that's exactly
    what Phase 5's citation enforcement will need to guard against).
    """
    result = chain.invoke({"question": query}, return_only_outputs=True)
    citations = extract_citations(result["source_documents"])
    return result["answer"], citations