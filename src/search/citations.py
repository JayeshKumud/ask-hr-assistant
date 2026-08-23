"""
Search: turns retrieved source Documents into structured Citation objects.

A Citation is the answer's "how do I verify this" mechanism — each one
points at exactly which document, page, and chunk (roughly a paragraph)
a piece of the answer was drawn from, plus the actual text, so a user
can check the claim without opening the source PDF.

Phase 5 (citation enforcement) will reuse extract_citations() to check
whether an answer's claims are actually supported by these chunks, rather
than trusting the LLM's self-reported "SOURCES:" line.
"""
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document


@dataclass(frozen=True)
class Citation:
    """One verifiable reference: a document, a location within it, and the
    exact text the answer relied on."""

    source: str  # filename, e.g. "NexaCore_..._Policy.pdf"
    page: Optional[int]  # 0-indexed page number, from PyPDFLoader metadata
    page_label: Optional[str]  # human-facing page number, e.g. "3"
    chunk_index: Optional[int]  # position of this chunk within its page
    snippet: str  # the chunk's actual text — what backs the answer

    def display_label(self) -> str:
        """Short human-readable label, e.g. 'NexaCore_..._Policy.pdf — page 3'."""
        if self.page_label:
            return f"{self.source} — page {self.page_label}"
        return self.source


def extract_citations(source_documents: List[Document]) -> List[Citation]:
    """
    Converts retrieved chunks into deduplicated Citation objects.

    Deduplicates by (source, page, chunk_index) so the same page/chunk
    location doesn't get listed twice even if the retriever returned
    overlapping chunks.
    """
    seen = set()
    citations: List[Citation] = []

    for doc in source_documents:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                page_label=doc.metadata.get("page_label"),
                chunk_index=doc.metadata.get("chunk_index"),
                snippet=doc.page_content.strip(),
            )
        )

    return citations


if __name__ == "__main__":
    # Quick manual check with fake Documents — no network/API calls
    # needed, just confirms dedup + display_label() behave correctly.
    sample_docs = [
        Document(
            page_content="Full-time employees receive 25 working days of annual leave.",
            metadata={"source": "policy.pdf", "page": 1, "page_label": "2", "chunk_index": 0},
        ),
        Document(
            page_content="Full-time employees receive 25 working days of annual leave.",
            metadata={"source": "policy.pdf", "page": 1, "page_label": "2", "chunk_index": 0},
        ),  # exact duplicate on purpose — should collapse to one citation
        Document(
            page_content="Sick pay is provided at 100% of base salary for 10 days.",
            metadata={"source": "policy.pdf", "page": 2, "page_label": "3", "chunk_index": 1},
        ),
    ]

    found_citations = extract_citations(sample_docs)
    print(f"{len(sample_docs)} source documents -> {len(found_citations)} deduplicated citations")
    for c in found_citations:
        print(f"- {c.display_label()}: {c.snippet[:60]}...")