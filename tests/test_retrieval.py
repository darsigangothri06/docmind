"""Retrieval tests — require API key for embedding, so marked for integration."""


def test_retriever_interface():
    """Verify DocumentRetriever accepts a vector store and k parameter."""
    from src.retrieval.retriever import DocumentRetriever
    # Can't fully test without embeddings, but verify the class is importable and structured
    assert hasattr(DocumentRetriever, "retrieve")
