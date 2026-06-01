from __future__ import annotations

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document


class DocumentLoader:
    """Loads documents from supported file formats (PDF, MD, TXT)."""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }

    def load(self, file_path: str) -> list[Document]:
        ext = Path(file_path).suffix.lower()
        loader_cls = self.LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported format: {ext}. Supported: {list(self.LOADER_MAP.keys())}")
        return loader_cls(file_path).load()

    def load_directory(self, dir_path: str) -> list[Document]:
        docs = []
        for file in Path(dir_path).rglob("*"):
            if file.suffix.lower() in self.LOADER_MAP:
                docs.extend(self.load(str(file)))
        return docs
