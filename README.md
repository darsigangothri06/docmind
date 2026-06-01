# DocMind — RAG Knowledge Assistant

A Retrieval-Augmented Generation system for conversational Q&A over custom document collections. Upload PDFs, markdown, or text files — ask questions in natural language and get grounded answers with source citations.

## Features

- **Multi-format ingestion** — PDF, Markdown, Plain Text
- **Smart chunking** — Recursive character splitting with overlap
- **Vector search** — ChromaDB with MMR reranking
- **Multi-provider LLM** — OpenAI (GPT-4o-mini) or Google (Gemini 2.5 Flash)
- **Source citations** — Every answer references its source document + page
- **Evaluation pipeline** — 4 metrics: Faithfulness, Relevance, Precision, Recall
- **REST API** — FastAPI with OpenAPI docs
- **Chat UI** — Streamlit interface with collection management

## Quick Start

```bash
# 1. Virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Start API server
uvicorn src.api.main:app --reload --port 8000

# 4. Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

- API docs: http://localhost:8000/docs
- Chat UI: http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/collections/{name}/upload` | Upload documents to a collection |
| POST | `/api/collections/{name}/query` | Query a collection |
| GET | `/api/collections` | List all collections |
| DELETE | `/api/collections/{name}` | Delete a collection |
| POST | `/api/evaluate` | Run evaluation pipeline |
| GET | `/health` | Health check |

## Configuration

Set API keys via the Streamlit UI sidebar or environment variables (see `.env.example`).

## Tech Stack

- Python 3.11+ | LangChain 0.3.x | ChromaDB | FastAPI | Streamlit
- Embeddings: OpenAI text-embedding-3-small / Google text-embedding-004
- LLM: OpenAI GPT-4o-mini / Google Gemini 2.5 Flash
