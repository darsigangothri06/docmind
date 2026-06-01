import tempfile
from pathlib import Path
from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import DocumentSplitter


def test_loader_supports_txt():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world. This is a test document for DocMind.")
        f.flush()
        docs = loader.load(f.name)
    assert len(docs) >= 1
    assert "Hello world" in docs[0].page_content


def test_loader_rejects_unsupported():
    loader = DocumentLoader()
    try:
        loader.load("test.xyz")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported" in str(e)


def test_splitter_chunks():
    from langchain_core.documents import Document
    splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
    docs = [Document(page_content="A" * 200, metadata={"source": "test"})]
    chunks = splitter.split(docs)
    assert len(chunks) > 1
