from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from askhr.core.config import settings


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Splits documents into chunks using the configured chunk size,
    overlap, and separators from Settings.

    Each resulting chunk gets a `chunk_index` in its metadata: its
    position among the chunks that came from the same page (0, 1, 2...).
    RecursiveCharacterTextSplitter already copies the parent document's
    metadata (source, page, page_label) onto every chunk it produces, but
    doesn't distinguish *which* chunk within that page a given piece of
    text came from — chunk_index adds that. Combined with page number,
    this is what lets a citation point at a specific chunk of text rather
    than just "somewhere in this page".
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=list(settings.chunk_separators),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    # ------------------------------------------------------------
    # Purpose:
    # Assign a sequential chunk_index to every text chunk generated
    # from each document page. The index starts at 0 for each page
    # and increments for every additional chunk belonging to the
    # same (source, page). This helps maintain ordering, improves
    # retrieval accuracy, and preserves page structure in RAG flows.
    # ------------------------------------------------------------

    # Split the input documents into smaller text chunks
    chunks: List[Document] = splitter.split_documents(docs)

    # Dictionary to track how many chunks have been created per page
    # Key format: (source, page) → count
    counters: Dict[Tuple, int] = {}

    # Iterate through all generated chunks
    for chunk in chunks:
        # Identify the page the chunk belongs to
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))

        # Assign a chunk_index starting from 0 for each page
        chunk.metadata["chunk_index"] = counters.get(key, 0)

        # Increment the counter so the next chunk gets the next index
        counters[key] = counters.get(key, 0) + 1

    return chunks


if __name__ == "__main__":
    # Manual check against the real policy PDF: confirms chunk_index
    # actually increments correctly within each page rather than just
    # being 0 everywhere (an easy off-by-reference bug with a shared
    # counter dict).
    from askhr.ingestion.document_loader import load_documents

    pages = load_documents()
    found_chunks = split_documents(pages)

    print(f"{len(pages)} pages -> {len(found_chunks)} chunks")
    for found_chunk in found_chunks[:6]:
        print(
            f"page={found_chunk.metadata.get('page')} "
            f"chunk_index={found_chunk.metadata.get('chunk_index')} "
            f"text={found_chunk.page_content[:50]!r}"
        )