# DocMind — RAG Knowledge Assistant

> Build spec for Cursor/Claude agents. Follow this document to build and deploy the project.

## Purpose

A Retrieval-Augmented Generation system for conversational Q&A over custom document collections. Users upload documents (PDFs, markdown, text), the system chunks and embeds them into a vector store, and answers questions with source citations. Includes an evaluation pipeline for measuring response quality across 4 dimensions.

**Problem it solves:** Teams drown in documentation — wikis, PDFs, READMEs, specs. DocMind lets you upload everything and ask questions in natural language, getting grounded answers with exact source references.

## Architecture

```
INGESTION FLOW:
  Documents (PDF/MD/TXT) → Document Loader → Text Splitter → Embedding Model → ChromaDB

QUERY FLOW:
  User Question → Query Embedding → Vector Search (MMR) → Context Assembly → LLM → Answer + Sources

EVALUATION FLOW:
  Test Dataset → RAG Pipeline → Metrics (Faithfulness, Relevance, Precision, Recall) → Score Report
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| RAG orchestration | LangChain 0.3.x |
| Vector store | ChromaDB (persistent, local) |
| Embeddings | OpenAI `text-embedding-3-small` OR Google `text-embedding-004` |
| LLM | OpenAI `gpt-4o-mini` OR Google `gemini-2.5-flash` (user-configured) |
| API server | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Document parsing | PyPDF2, Unstructured |
| Python version | 3.11+ |

## Directory Structure

```
docmind/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Environment + provider config
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # Document loaders (PDF, MD, TXT)
│   │   ├── splitter.py           # Text chunking
│   │   └── embedder.py           # Embedding + vector store management
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # Vector search + MMR reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── chain.py              # RAG chain (retrieval + generation)
│   │   ├── prompts.py            # System/user prompt templates
│   │   └── models.py             # Response dataclasses
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py          # Evaluation orchestrator
│   │   ├── metrics.py            # 4 evaluation metrics
│   │   └── dataset.py            # Test dataset loader
│   └── api/
│       ├── __init__.py
│       ├── main.py               # FastAPI app
│       ├── routes.py             # API endpoints
│       └── schemas.py            # Request/response models
├── ui/
│   └── app.py                    # Streamlit chat interface
├── data/
│   ├── documents/                # Upload directory
│   └── eval/
│       └── test_dataset.json     # Evaluation test cases
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_chain.py
│   └── test_evaluation.py
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

## Implementation Guide

### 1. Document Ingestion (`src/ingestion/`)

**loader.py:**
```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoader:
    """Loads documents from supported file formats."""

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }

    def load(self, file_path: str) -> list[Document]:
        ext = Path(file_path).suffix.lower()
        loader_cls = self.LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported: {ext}. Supported: {list(self.LOADER_MAP.keys())}")
        return loader_cls(file_path).load()

    def load_directory(self, dir_path: str) -> list[Document]:
        docs = []
        for file in Path(dir_path).rglob("*"):
            if file.suffix.lower() in self.LOADER_MAP:
                docs.extend(self.load(str(file)))
        return docs
```

**splitter.py:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)
```

**embedder.py:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

class VectorStoreManager:
    def __init__(self, provider: str = "openai", persist_dir: str = "./chroma_db",
                 api_key: str = ""):
        self.embeddings = self._get_embeddings(provider, api_key)
        self.persist_dir = persist_dir

    def _get_embeddings(self, provider: str, api_key: str):
        if provider == "openai":
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=api_key
        )

    def create_collection(self, documents: list, collection: str) -> Chroma:
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
```

### 2. Retrieval (`src/retrieval/retriever.py`)

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

class DocumentRetriever:
    def __init__(self, vector_store: Chroma, k: int = 5):
        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.7},
        )

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)
```

### 3. RAG Chain (`src/generation/`)

**chain.py:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .prompts import SYSTEM_PROMPT
from .models import RAGResponse

class RAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.chain = (
            {"context": retriever.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question: str) -> RAGResponse:
        docs = self.retriever.retrieve(question)
        answer = self.chain.invoke(question)
        return RAGResponse(
            answer=answer,
            sources=[
                {"content": d.page_content[:200], "metadata": d.metadata}
                for d in docs
            ],
        )

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, "
            f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )
```

**prompts.py:**
```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer from the provided context. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
2. Cite your sources by referencing the document name and page number when available.
3. Be concise and accurate.
4. If the question is ambiguous, ask for clarification.

Context:
{context}"""
```

**models.py:**
```python
from dataclasses import dataclass

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
```

### 4. Evaluation (`src/evaluation/`)

**metrics.py:**
```python
class RAGEvaluator:
    """Evaluate RAG pipeline quality across 4 dimensions."""

    def __init__(self, llm):
        self.llm = llm

    def faithfulness(self, answer: str, context: str) -> float:
        """Is the answer grounded in context? (0.0 - 1.0)
        Extracts claims from answer, checks each against context."""
        prompt = f"""Extract factual claims from this answer, then check if each is supported by the context.

Answer: {answer}
Context: {context}

Output JSON: {{"claims": [{{"claim": "...", "supported": true/false}}], "score": 0.0-1.0}}"""
        result = self.llm.invoke(prompt)
        # Parse and return score
        pass

    def answer_relevance(self, answer: str, question: str) -> float:
        """Does the answer address the question? (0.0 - 1.0)
        Generates questions from answer, measures cosine similarity with original."""
        pass

    def context_precision(self, contexts: list[str], question: str) -> float:
        """Are the retrieved chunks relevant? (0.0 - 1.0)
        Checks what proportion of retrieved chunks are useful for answering."""
        pass

    def context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """Did we retrieve all necessary context? (0.0 - 1.0)
        Checks if ground truth claims are covered by retrieved context."""
        pass

    def evaluate(self, question: str, answer: str, contexts: list[str],
                 ground_truth: str | None = None) -> dict:
        context_text = "\n".join(contexts)
        scores = {
            "faithfulness": self.faithfulness(answer, context_text),
            "answer_relevance": self.answer_relevance(answer, question),
            "context_precision": self.context_precision(contexts, question),
        }
        if ground_truth:
            scores["context_recall"] = self.context_recall(contexts, ground_truth)
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores
```

**dataset.py:**
```python
import json
from pathlib import Path

class EvalDataset:
    """Load evaluation test cases from JSON."""

    def load(self, path: str) -> list[dict]:
        data = json.loads(Path(path).read_text())
        return data  # [{question, ground_truth, expected_contexts?}]
```

**Test dataset format (`data/eval/test_dataset.json`):**
```json
[
    {
        "question": "What is the main purpose of the system?",
        "ground_truth": "The system is designed to...",
        "expected_sources": ["doc1.pdf"]
    }
]
```

### 5. FastAPI Server (`src/api/`)

**routes.py:**
```python
from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/collections/{name}/upload")
async def upload_documents(name: str, files: list[UploadFile] = File(...)):
    """Upload documents to a named collection."""
    pass

@router.post("/collections/{name}/query")
async def query_collection(name: str, request: QueryRequest):
    """Ask a question against a collection."""
    pass

@router.get("/collections")
async def list_collections():
    """List all document collections with stats."""
    pass

@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection and its data."""
    pass

@router.post("/evaluate")
async def run_evaluation(request: EvalRequest):
    """Run evaluation pipeline on test dataset."""
    pass
```

### 6. Streamlit UI (`ui/app.py`)

Layout:
- **Sidebar:**
  - Settings: LLM provider, API key, embedding provider
  - Collection management: create, select, delete
  - File upload (multi-file)
  - Collection stats (doc count, chunk count)
- **Main area:**
  - Chat interface with message history
  - Each answer has expandable "Sources" section
  - "Evaluate" tab with metrics dashboard

```python
import streamlit as st

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["gemini", "openai"])
    api_key = st.text_input("API Key", type="password")

    st.divider()
    st.header("Collections")
    # Collection selector, upload, stats

tab_chat, tab_eval = st.tabs(["Chat", "Evaluate"])

with tab_chat:
    st.title("DocMind")
    st.caption("Chat with your documents.")
    # Chat interface with st.chat_message

with tab_eval:
    st.title("Evaluation")
    # Upload test dataset, run evaluation, show metrics table
```

## Environment Variables

```env
# LLM (configurable via UI settings page)
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_MODEL=gemini-2.5-flash
EMBEDDING_PROVIDER=gemini

# Storage
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./data/documents

# Server
API_HOST=0.0.0.0
API_PORT=8000
```

## Dependencies (`requirements.txt`)

```
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-google-genai>=2.0.0
langchain-chroma>=0.2.0
langchain-community>=0.3.0
chromadb>=0.5.0
fastapi>=0.115.0
uvicorn>=0.30.0
streamlit>=1.40.0
pypdf2>=3.0.0
unstructured>=0.16.0
python-dotenv>=1.0.0
pydantic>=2.0.0
python-multipart>=0.0.9
```

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/darsigangothri06/docmind.git
cd docmind

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Environment (optional — can use UI settings instead)
cp .env.example .env

# 5. Start API server
uvicorn src.api.main:app --reload --port 8000

# 6. Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501

# Open:
# API docs: http://localhost:8000/docs
# Chat UI: http://localhost:8501
```

## Testing

```bash
pytest tests/ -v
```

## Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## GitHub Repository Setup

> **CRITICAL — READ BEFORE ANY GIT OPERATION**
>
> This project MUST be pushed to the **personal** GitHub account ONLY.
> - **CORRECT account:** `darsigangothri06` (gangothri.darsi@gmail.com)
> - **DO NOT USE:** `gangothri-bryt` / `gangothri@bryt.in` — this is the company work account. NEVER push personal projects to the work account.
> - **DO NOT modify global git config** (`--global`). Only set LOCAL config inside this repo.
> - **VERIFY before every push:** Run `git config user.name && git config user.email` and confirm it shows `darsigangothri06` / `gangothri.darsi@gmail.com`. If it doesn't, STOP and fix it.

```bash
# 1. Create repo on GitHub first
gh repo create darsigangothri06/docmind --public --description "RAG knowledge assistant with evaluation pipeline"

# 2. Initialize local repo
git init

# 3. SET LOCAL GIT IDENTITY (NOT --global)
git config user.name "darsigangothri06"
git config user.email "gangothri.darsi@gmail.com"

# 4. VERIFY identity before proceeding
git config user.name   # Must show: darsigangothri06
git config user.email  # Must show: gangothri.darsi@gmail.com

# 5. Add remote and push
git remote add origin https://github.com/darsigangothri06/docmind.git
git add .
git commit -m "feat: DocMind — RAG knowledge assistant with evaluation pipeline"
git push -u origin main
```

If `gh` CLI is authenticated as the work account, authenticate personal account first:
```bash
gh auth login  # Choose: github.com → HTTPS → Login with browser → authenticate as darsigangothri06
```
