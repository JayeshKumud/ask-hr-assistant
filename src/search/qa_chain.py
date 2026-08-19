from typing import Tuple, List

from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.chains.qa_with_sources.loading import load_qa_with_sources_chain

from core.config import settings
from search.prompts import PROMPT, EXAMPLE_PROMPT


def build_qa_chain(llm, vector_store) -> RetrievalQAWithSourcesChain:
    """
    Builds the retrieval-augmented QA chain from an LLM and a vector store.
    """
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=PROMPT,
        document_prompt=EXAMPLE_PROMPT,
    )
    return RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vector_store.as_retriever(search_kwargs={"k": settings.top_k}),
        reduce_k_below_max_tokens=True,
        max_tokens_limit=settings.max_tokens_limit,
        return_source_documents=True,
    )


def generate_answer(chain: RetrievalQAWithSourcesChain, query: str) -> Tuple[str, List[str]]:
    """
    Runs a query through the QA chain, returns (answer, list of source URLs).
    """
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = [doc.metadata["source"] for doc in result["source_documents"]]
    return result["answer"], sources
