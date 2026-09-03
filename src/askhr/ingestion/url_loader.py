from typing import List

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document

# Some sites (CNBC included) return an "Access Denied" page instead of the
# article when the request doesn't look like it's coming from a real
# browser. Without this, UnstructuredURLLoader happily "succeeds" and
# scrapes the denial page's text instead of the actual article.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def load_urls(urls: List[str]) -> List[Document]:
    """
    Scrapes and loads content from a list of URLs.
    """
    if not urls:
        raise ValueError("No URLs provided to load_urls()")

    loader = UnstructuredURLLoader(
        urls=urls,
        headers=_BROWSER_HEADERS,
    )

    data = loader.load()
    for doc in data:
        if "access denied" in doc.page_content.lower() and len(doc.page_content) < 500:
            raise RuntimeError(f"Blocked or empty content from {doc.metadata.get('source')}")

    return data
