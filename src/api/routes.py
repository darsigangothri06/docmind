from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from .schemas import QueryRequest, QueryResponse, CollectionInfo, EvalRequest, EvalResponse
from ..config import get_settings, ensure_dirs
from ..ingestion.loader import DocumentLoader
from ..ingestion.splitter import DocumentSplitter
from ..ingestion.embedder import VectorStoreManager
from ..retrieval.retriever import DocumentRetriever
from ..generation.chain import RAGChain, get_llm
from ..evaluation.evaluator import EvaluationOrchestrator

router = APIRouter()


@router.post("/collections/{name}/upload", summary="Upload documents to a collection")
async def upload_documents(
    name: str,
    files: list[UploadFile] = File(...),
    provider: str = Form("gemini"),
    api_key: str = Form(""),
):
    """Upload documents to a named collection. Creates collection if it doesn't exist."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    settings = get_settings()
    ensure_dirs(settings)

    loader = DocumentLoader()
    splitter = DocumentSplitter()
    all_docs = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            ext = Path(file.filename).suffix.lower()
            if ext not in loader.LOADER_MAP:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file: {file.filename}. Supported: {list(loader.LOADER_MAP.keys())}",
                )
            tmp_path = Path(tmp_dir) / file.filename
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            docs = loader.load(str(tmp_path))
            all_docs.extend(docs)

    if not all_docs:
        raise HTTPException(status_code=400, detail="No documents could be loaded from uploaded files")

    chunks = splitter.split(all_docs)
    vsm = VectorStoreManager(provider=provider, persist_dir=settings.chroma_persist_dir, api_key=api_key)

    try:
        vsm.create_collection(chunks, collection=name)
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            raise HTTPException(status_code=401, detail=f"Invalid API key: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {error_msg}")

    return {
        "message": f"Uploaded {len(files)} file(s) to collection '{name}'",
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
    }


@router.post("/collections/{name}/query", response_model=QueryResponse, summary="Query a collection")
async def query_collection(name: str, request: QueryRequest):
    """Ask a question against a collection and get an answer with sources."""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    settings = get_settings()
    vsm = VectorStoreManager(
        provider=request.provider, persist_dir=settings.chroma_persist_dir, api_key=request.api_key
    )

    try:
        vector_store = vsm.load_collection(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found: {e}")

    retriever = DocumentRetriever(vector_store)
    llm = get_llm(request.provider, request.api_key, request.model)
    chain = RAGChain(retriever, llm)

    try:
        response = chain.query(request.question)
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg or "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            raise HTTPException(status_code=401, detail=f"Invalid API key: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")
    return QueryResponse(answer=response.answer, sources=response.sources)


@router.get("/collections", response_model=list[CollectionInfo], summary="List all collections")
async def list_collections(provider: str = "gemini", api_key: str = ""):
    """List all document collections with document counts."""
    settings = get_settings()
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        return []

    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collections = client.list_collections()
        result = []
        for col in collections:
            c = client.get_collection(col.name)
            result.append(CollectionInfo(name=col.name, document_count=c.count()))
        return result
    except Exception:
        return []


@router.delete("/collections/{name}", summary="Delete a collection")
async def delete_collection(name: str):
    """Delete a collection and its stored data."""
    settings = get_settings()
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.delete_collection(name)
        return {"message": f"Collection '{name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not delete collection '{name}': {e}")


@router.post("/evaluate", response_model=EvalResponse, summary="Run evaluation pipeline")
async def run_evaluation(request: EvalRequest):
    """Run evaluation pipeline on a test dataset against a collection."""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    settings = get_settings()
    vsm = VectorStoreManager(
        provider=request.provider, persist_dir=settings.chroma_persist_dir, api_key=request.api_key
    )

    try:
        vector_store = vsm.load_collection(request.collection)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found: {e}")

    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    retriever = DocumentRetriever(vector_store)
    llm = get_llm(request.provider, request.api_key, request.model)
    chain = RAGChain(retriever, llm)

    orchestrator = EvaluationOrchestrator(chain, llm)
    result = orchestrator.run(str(dataset_path))

    return EvalResponse(results=result["results"], average_scores=result["average_scores"])
