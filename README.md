# DocMind — RAG Knowledge Assistant

> Upload documents. Ask questions. Get grounded answers with source citations. Evaluate retrieval quality with a 4-metric pipeline.

**[Live Demo](https://docmind-ui-yltt.onrender.com)**

---

## What It Does

DocMind is a Retrieval-Augmented Generation (RAG) system for conversational Q&A over your own document collections. Upload PDFs, markdown, or text files — ask questions in natural language — get answers grounded in your documents with exact source references.

**What makes it different from a basic RAG demo:**
- **Source citations** — every answer references the source document + page number
- **Evaluation pipeline** — 4 automated metrics to measure retrieval and answer quality (not just "does it look right?")
- **Multi-provider** — switch between OpenAI and Gemini for both embeddings and generation
- **Collection management** — organize documents into named collections, each with its own vector index

## Architecture

```
INGESTION:
  Documents (PDF/MD/TXT)
    → Document Loader (format detection)
    → Recursive Text Splitter (1000 chars, 200 overlap)
    → Embedding Model (OpenAI or Gemini)
    → ChromaDB (persistent vector store, per-collection)

QUERY:
  User Question
    → Query Embedding
    → Vector Search (MMR reranking, k=5, fetch_k=20)
    → Context Assembly (source metadata preserved)
    → LLM (GPT-4o-mini or Gemini 2.5 Flash)
    → Answer + Source Citations

EVALUATION:
  Test Dataset (question + ground truth pairs)
    → RAG Pipeline (retrieve + generate)
    → 4 Metrics: Faithfulness | Relevance | Context Precision | Context Recall
    → Score Report
```

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Chunking | **RecursiveCharacterTextSplitter** (1000/200) | Respects document structure (paragraphs → sentences → words), overlap prevents context loss at boundaries |
| Retrieval | **MMR reranking** | Balances relevance with diversity — prevents returning 5 nearly-identical chunks |
| Vector DB | **ChromaDB** (persistent, local) | Zero infrastructure overhead, good enough for single-node deployments, collection-based organization |
| Evaluation | **LLM-as-judge** (4 metrics) | Automated quality measurement without human annotation — Faithfulness, Relevance, Precision, Recall |
| Multi-provider | **OpenAI + Gemini** | Embedding and LLM provider configurable at runtime — cost/latency flexibility |

## The Evaluation Pipeline (Most Important Part)

Most RAG tutorials stop at "it returns an answer." DocMind measures **how good** that answer is:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? | Catches hallucination |
| **Answer Relevance** | Does the answer address the actual question? | Catches tangential responses |
| **Context Precision** | Are the retrieved chunks actually useful? | Measures retrieval quality |
| **Context Recall** | Did we retrieve all the necessary context? | Catches missed information |

## Tech Stack

`Python` `LangChain 0.3.x` `ChromaDB` `FastAPI` `Streamlit` `OpenAI` `Gemini`

## Quick Start

```bash
git clone https://github.com/darsigangothri06/docmind.git
cd docmind

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start API
uvicorn src.api.main:app --reload --port 8000

# Start UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

API keys are configured in the Streamlit UI sidebar (or via `.env` file).

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/collections/{name}/upload` | Upload documents to a collection |
| POST | `/api/collections/{name}/query` | Query a collection |
| GET | `/api/collections` | List all collections |
| DELETE | `/api/collections/{name}` | Delete a collection |
| POST | `/api/evaluate` | Run evaluation pipeline |
| GET | `/health` | Health check |

## Project Structure

```
docmind/
├── src/
│   ├── ingestion/
│   │   ├── loader.py       # Multi-format document loading (PDF, MD, TXT)
│   │   ├── splitter.py     # Recursive character text splitting
│   │   └── embedder.py     # Embedding + ChromaDB vector store management
│   ├── retrieval/
│   │   └── retriever.py    # MMR vector search with configurable k
│   ├── generation/
│   │   ├── chain.py        # RAG chain (retrieval + LLM generation)
│   │   ├── prompts.py      # System prompt with citation instructions
│   │   └── models.py       # Response dataclasses
│   ├── evaluation/
│   │   ├── evaluator.py    # Evaluation orchestrator
│   │   ├── metrics.py      # 4-metric RAG evaluation (Faithfulness, Relevance, Precision, Recall)
│   │   └── dataset.py      # Test dataset loader
│   └── api/
│       ├── main.py         # FastAPI app
│       ├── routes.py       # REST endpoints
│       └── schemas.py      # Pydantic request/response models
├── ui/
│   └── app.py              # Streamlit chat interface + evaluation dashboard
├── data/eval/
│   └── test_dataset.json   # Sample evaluation test cases
├── tests/
├── requirements.txt
├── Dockerfile
└── .env.example
```

## What I'd Improve

- **Hybrid search** — combine BM25 keyword search with vector search for better retrieval on exact-match queries
- **Document structure awareness** — use headings/sections as chunk boundaries instead of character counts
- **Conversation memory** — maintain chat history for multi-turn follow-up questions
- **Fine-tuned embeddings** — train domain-specific embedding models for specialized document collections

## Author

**Gangothri Darsi** — [GitHub](https://github.com/darsigangothri06) | [LinkedIn](https://www.linkedin.com/in/darsigangothri06) | [Portfolio](https://gangothridarsi.vercel.app)
