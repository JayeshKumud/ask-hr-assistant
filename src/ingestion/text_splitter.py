from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import settings


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Splits documents into chunks using the configured chunk size,
    overlap, and separators from Settings.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=list(settings.chunk_separators),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)
