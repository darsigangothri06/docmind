"""RAG chain tests — require API key for LLM calls, marked for integration."""


def test_rag_response_model():
    from src.generation.models import RAGResponse
    resp = RAGResponse(answer="test answer", sources=[{"content": "c", "metadata": {}}])
    assert resp.answer == "test answer"
    assert len(resp.sources) == 1


def test_get_llm_factory():
    from src.generation.chain import get_llm
    # Verify factory doesn't crash (actual calls need API key)
    llm = get_llm("gemini", "fake-key", "gemini-2.5-flash")
    assert llm is not None
