# NTT DATA RAG API - Hybrid Search with LangGraph

Production-ready RAG (Retrieval Augmented Generation) system with async architecture:

- **Hybrid Search**: Dense (Gemini) + Sparse (FastEmbed BM25) + RRF Fusion
- **Auto Year Extraction**: LLM automatically extracts years from questions
- **Async Architecture**: Non-blocking requests, high concurrency
- **Lifespan DI**: Singleton services, initialized once at startup
- **Service Layer**: Clean separation of business logic

## 🎯 Features

| Feature | Description |
|---------|-------------|
| ✅ Hybrid Search | Dense + Sparse + RRF fusion |
| ✅ Year Extraction | Auto-extract from queries |
| ✅ Async | `ainvoke()`, non-blocking |
| ✅ Lifespan DI | Services in `app.state` |
| ✅ Service Layer | RAGService business logic |
| ✅ Custom Exceptions | LLMException, VectorStoreException |
| ✅ Retry Logic | Tenacity with exponential backoff |
| ✅ Docker Ready | Compose for prod/dev |

## 🚀 Quick Start

### 1. Setup
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure
```bash
copy .env.example .env
# Edit .env with your Google API key
```

### 3. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run API
```bash
uvicorn src.main:app --reload
```

Visit: http://localhost:8000/docs

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ask` | POST | RAG query |
| `/api/v1/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

### Example Request
```bash
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/ask" `
  -Method POST -ContentType "application/json" `
  -Body '{"question": "2023 NTT DATA sustainability"}'
```

### Example Response
```json
{
  "answer": "NTT DATA's 2023 sustainability report...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "rewritten_question": "NTT DATA sustainability strategy 2023",
  "years_extracted": [2023]
}
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Lifespan                          │    │
│  │  Services initialized ONCE at startup:              │    │
│  │  • LLMService      • VectorStore                    │    │
│  │  • EmbeddingService • RAGService                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                    stored in app.state                       │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   API Layer                          │    │
│  │  /ask ──► dependencies.py ──► RAGService.ask()      │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                    await ainvoke()                           │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 LangGraph Workflow                   │    │
│  │  [Rewrite] ──► [Retrieve] ──► [Generate]            │    │
│  │   • Turkish→EN   • Hybrid search  • LLM response    │    │
│  │   • Year extract • RRF fusion     • Context-based   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **Dependency Injection** | Lifespan + `app.state` |
| **Service Layer** | `RAGService` orchestrates workflow |
| **Interface Segregation** | `BaseLLMService`, `BaseVectorStore` |
| **Async/Await** | `graph.ainvoke()` for non-blocking |

## 📁 Project Structure

```
src/
├── api/
│   ├── dependencies.py      # DI: get_rag_service()
│   └── v1/endpoints/
│       ├── rag.py           # POST /ask
│       └── health.py        # GET /health
├── core/
│   ├── config.py            # Pydantic Settings
│   ├── interfaces.py        # ABCs
│   ├── exceptions.py        # Custom exceptions
│   ├── prompts.py           # YAML prompt loader
│   ├── logging_config.py    # Logging configuration
│   └── state.py             # GraphState TypedDict
├── models/
│   └── schemas.py           # Pydantic models for API
├── services/
│   ├── rag_service.py       # Business logic layer
│   ├── llm.py               # Gemini with retry
│   ├── embeddings.py        # Dense + Sparse
│   └── vector_store.py      # Qdrant hybrid search
├── workflows/
│   ├── nodes/               # Rewrite, Retrieve, Generate
│   └── graph.py             # LangGraph assembly
├── prompts/
│   └── prompts.yaml         # LLM prompts
└── main.py                  # Lifespan, app creation

scripts/
├── convert_pdfs.py          # PDF to Markdown converter
└── ingest_data.py           # Data ingestion pipeline

notebooks/
├── chunking_experiments.ipynb  # Chunking strategy experiments
├── data_analyze.ipynb         # Data analysis and exploration
└── ocr_test.ipynb             # OCR testing
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

## 📊 Data Ingestion

The project includes scripts for data preparation and ingestion:

### PDF Conversion
Convert PDF documents to Markdown format:
```bash
python scripts/convert_pdfs.py --input <pdf_file_or_dir> --output <output_dir>
```

### Data Ingestion
Ingest processed documents into the vector database:
```bash
python scripts/ingest_data.py
```

## 📓 Notebooks

Experimental notebooks for analysis and testing:

| Notebook | Description |
|----------|-------------|
| `chunking_experiments.ipynb` | Test different chunking strategies |
| `data_analyze.ipynb` | Analyze and explore document data |
| `ocr_test.ipynb` | Test OCR capabilities |

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | Required | Google API key for LLM |
| `EMBEDDING_API_KEY` | Required | Google API key for embeddings |
| `QDRANT_URL` | http://localhost:6333 | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | ntt_hybrid_experiment | Vector collection name |
| `LLM_MODEL` | gemini-2.5-flash | LLM model name |
| `EMBEDDING_MODEL` | models/embedding-001 | Embedding model name |
| `LLM_TEMPERATURE` | 0.7 | LLM temperature setting |
| `RAG_K` | 5 | Number of results to retrieve |
| `LOG_LEVEL` | INFO | Logging level |
| `APP_HOST` | 127.0.0.1 | API server host |
| `APP_PORT` | 8000 | API server port |

## 🐳 Docker

```bash
# Development (hot reload)
docker-compose -f docker-compose.dev.yaml up

# Production
docker-compose up -d
```

## 📝 License

Internal NTT DATA project

---

**Status**: ✅ Production Ready | **Tests**: All Passing | **Architecture**: Async + LangGraph
