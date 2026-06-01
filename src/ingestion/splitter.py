from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentSplitter:
    """Chunks documents using recursive character splitting with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)
