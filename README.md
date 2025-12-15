# NTT DATA RAG API

Production-ready Retrieval Augmented Generation system with hybrid search and LangGraph workflow.

## Features

- **Hybrid Search** - Dense (Gemini) + Sparse (BM25) + RRF fusion
- **Smart Query Processing** - Automatic year extraction and query rewriting
- **Async Architecture** - Non-blocking, high-concurrency design
- **Provider Flexibility** - Switch between LLM providers via configuration
- **Clean Architecture** - Factory pattern, dependency injection, layered design

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)

### Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your credentials
LLM_API_KEY=your_api_key_here
EMBEDDING_API_KEY=your_api_key_here
```

### Start Services

```bash
# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant

# Start API server
python -m uvicorn src.main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for interactive API documentation.

---

## API Usage

### Ask a Question

```bash
POST http://localhost:8000/api/v1/ask
Content-Type: application/json

{
  "question": "What are NTT DATA's 2023 sustainability initiatives?"
}
```

### Response

```json
{
  "answer": "NTT DATA's 2023 sustainability report highlights...",
  "sources": ["sustainability_report_2023.pdf", "annual_report_2023.pdf"],
  "rewritten_question": "NTT DATA sustainability strategy and initiatives 2023",
  "years_extracted": [2023]
}
```

### Health Check

```bash
GET http://localhost:8000/api/v1/health
```

---

## Architecture

### Request Flow

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Application                 │
│                                                      │
│  Startup:                                           │
│    • Initialize LLM service (Gemini/OpenAI)        │
│    • Initialize vector store (Qdrant)              │
│    • Initialize RAG service                         │
│                                                      │
│  Request: /api/v1/ask                               │
│    ├─ Validate input                                │
│    ├─ RAG Service                                   │
│    │   └─ LangGraph Workflow                        │
│    │       ├─ [Rewrite] Query optimization          │
│    │       ├─ [Retrieve] Hybrid search              │
│    │       └─ [Generate] Answer with context        │
│    └─ Return response                               │
└─────────────────────────────────────────────────────┘
```

### LangGraph Workflow

1. **Rewrite Node** - Improves query and extracts temporal information
2. **Retrieve Node** - Performs hybrid search (dense + sparse embeddings)
3. **Generate Node** - Generates answer using retrieved context

---

## Project Structure

```
src/
├── api/                    HTTP Layer
│   ├── dependencies.py      Dependency injection providers
│   └── v1/endpoints/
│       ├── rag.py           RAG endpoint
│       └── health.py        Health check
│
├── container/              Dependency Injection
│   ├── container.py         Service container
│   └── bootstrap.py         Service initialization
│
├── core/                   Domain Layer
│   ├── config.py            Application settings
│   ├── enums.py             Provider enums
│   ├── interfaces.py        Base interfaces
│   ├── exceptions.py        Custom exceptions
│   └── state.py             Workflow state
│
├── services/               Business Logic
│   ├── rag_service.py       RAG orchestration
│   ├── llm/
│   │   ├── factory.py       LLM factory
│   │   ├── gemini.py        Gemini implementation
│   │   └── openai.py        OpenAI implementation
│   ├── embeddings/
│   │   ├── factory.py       Embedding factory
│   │   ├── gemini.py        Dense embeddings
│   │   └── fastembed.py     Sparse embeddings (BM25)
│   └── vector_stores/
│       ├── factory.py       Vector store factory
│       └── qdrant.py        Qdrant implementation
│
├── workflows/              LangGraph Workflows
│   ├── graph.py             Workflow orchestration
│   └── nodes/
│       ├── rewrite.py       Query rewriting
│       ├── retrieve.py      Document retrieval
│       └── generate.py      Answer generation
│
├── prompts/                Prompt Templates
│   ├── prompts.py           Prompt manager
│   └── prompts.yaml         LLM prompts
│
└── main.py                 Application entry point
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | LLM provider (`gemini` or `openai`) |
| `LLM_API_KEY` | Required | API key for LLM |
| `LLM_MODEL` | `gemini-2.0-flash-exp` | LLM model name |
| `LLM_TEMPERATURE` | `0.7` | LLM temperature |
| `EMBEDDING_API_KEY` | Required | API key for embeddings |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `ntt_hybrid` | Collection name |
| `RAG_K` | `5` | Number of documents to retrieve |
| `LOG_LEVEL` | `INFO` | Logging level |

### Switch LLM Provider

To use OpenAI instead of Gemini:

```bash
# .env
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-key
LLM_MODEL=gpt-4o
```

No code changes required - the factory pattern handles provider selection automatically.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/core/ -v           # Core components
pytest tests/unit/workflows/ -v      # Workflow nodes
pytest tests/unit/services/ -v       # Services
pytest tests/integration/ -v         # API integration

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Data Ingestion

### Convert PDFs to Markdown

```bash
python scripts/convert_pdfs.py --input <pdf_file_or_dir> --output <output_dir>
```

### Ingest Documents

```bash
python scripts/ingest_data.py
```

The ingestion pipeline:
1. Reads markdown documents
2. Chunks text with metadata preservation
3. Generates dense and sparse embeddings
4. Stores in Qdrant with hybrid search support

---

## Docker Deployment

### Development

```bash
docker-compose -f docker-compose.dev.yaml up
```

Features hot reload for development.

### Production

```bash
docker-compose up -d
```

Runs with optimized settings for production.

---

## Technical Details

### Hybrid Search

- **Dense Embeddings** - Gemini text-embedding-004 for semantic search
- **Sparse Embeddings** - FastEmbed BM25 for keyword matching
- **Fusion** - Reciprocal Rank Fusion (RRF) combines both approaches

### Design Patterns

- **Factory Pattern** - Flexible provider switching without code changes
- **Container Pattern** - Centralized service lifecycle management
- **Dependency Injection** - Services injected via FastAPI's `Depends()`
- **Repository Pattern** - Vector store abstraction
- **Strategy Pattern** - Different LLM/embedding implementations

### Performance

- **Async/Await** - Non-blocking I/O throughout
- **Connection Pooling** - Efficient resource usage
- **Singleton Services** - Initialized once at startup
- **Lazy Loading** - Services created only when needed

---

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

### Adding a New LLM Provider

1. Create implementation in `src/services/llm/your_provider.py`
2. Implement `BaseLLMService` interface
3. Add provider to `LLMProvider` enum in `src/core/enums.py`
4. Register in `LLMFactory.create()` in `src/services/llm/factory.py`

Example:

```python
# src/services/llm/anthropic.py
class AnthropicLLMService(BaseLLMService):
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        # Implementation
        pass
    
    async def generate(self, prompt: str) -> str:
        # Implementation
        pass

# src/services/llm/factory.py
elif provider == LLMProvider.ANTHROPIC:
    return AnthropicLLMService(**kwargs)
```

---

## Monitoring

### Logs

Structured logging with different levels:

```python
# View logs
tail -f logs/app.log

# Filter by level
grep "ERROR" logs/app.log
```

### Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

Returns service status and dependencies.

---

**Status**: Production Ready  
**API**: FastAPI + LangGraph  
**Search**: Hybrid (Dense + Sparse)  
**Architecture**: Clean, Testable, Scalable
