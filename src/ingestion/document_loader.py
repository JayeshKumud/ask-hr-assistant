"""
Ingestion: loads local policy documents (PDF) instead of scraping URLs.

Replaces ingestion/url_loader.py. The old file scraped article text from
web URLs for the real-estate research use case; this system's source of
truth is a fixed set of company policy documents living on disk under
settings.policies_dir.

Uses LangChain's DirectoryLoader (rather than a hand-rolled directory walk)
to do the actual file discovery + loading, so the file-type -> loader-class
mapping and the load() call itself are both LangChain's responsibility.

PDF only for now — HTML support can be added back the same way (another
DirectoryLoader call + a dedup step) if/when it's needed.

Each loaded Document keeps page-level metadata (page number, source
filename) — this is what Phase 2's citation feature will point back to,
so it's worth getting right now rather than bolting it on later.
"""
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document

from core.config import settings

logger = logging.getLogger(__name__)


def load_documents(directory: Path = None) -> List[Document]:
    """
    Loads every PDF under `directory` (default: settings.policies_dir)
    via LangChain's DirectoryLoader + PyPDFLoader, and returns them as a
    flat list of Documents — one per page, with metadata["page"] set
    (the granularity citations need).
    """
    directory = directory or settings.policies_dir

    if not directory.exists():
        raise FileNotFoundError(f"Policy documents directory not found: {directory}")

    logger.info("Loading PDFs from %s", directory)

    # noinspection PyTypeChecker
    loader = DirectoryLoader(
        str(directory),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
    )
    documents = loader.load()

    # Normalize source metadata to just the filename (not the full local
    # disk path) — a full "C:\My Personal\Projects\..." path is not
    # something you want showing up in a citation shown to a user.
    for doc in documents:
        doc.metadata["source"] = Path(doc.metadata["source"]).name

    if not documents:
        raise RuntimeError(f"No PDF documents found in {directory}")

    return documents


if __name__ == "__main__":
    # Quick manual check: run this file directly to confirm the loader
    # works against your real policy documents before wiring it into the
    # pipeline. Prints how many Documents came back and a preview of the
    # first one's content + metadata.
    logging.basicConfig(level=logging.INFO)

    docs = load_documents()
    print(f"Loaded {len(docs)} document(s)/page(s)")
    print(f"First doc metadata: {docs[0].metadata}")
    print(f"First doc preview:\n{docs[0].page_content[:300]}")