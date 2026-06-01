from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages embedding generation and ChromaDB vector store operations."""

    def __init__(self, provider: str = "gemini", persist_dir: str = "./chroma_db",
                 api_key: str = ""):
        self.embeddings = self._get_embeddings(provider, api_key)
        self.persist_dir = persist_dir

    def _get_embeddings(self, provider: str, api_key: str):
        if provider == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )

    def create_collection(self, documents: list[Document], collection: str) -> Chroma:
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=collection,
        )

    def load_collection(self, collection: str) -> Chroma:
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=collection,
        )

    def list_collections(self) -> list[str]:
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        return [c.name for c in client.list_collections()]

    def delete_collection(self, collection: str):
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        client.delete_collection(collection)
