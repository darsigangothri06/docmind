from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document


class DocumentRetriever:
    """Vector search with MMR (Maximal Marginal Relevance) reranking."""

    def __init__(self, vector_store: Chroma, k: int = 5):
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.7},
        )

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
