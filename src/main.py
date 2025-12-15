"""FastAPI application entry point with lifespan management."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from src.core import setup_logging, PromptManager, settings
from src.services import (
    GeminiLLMService, 
    GeminiEmbeddingService, 
    FastEmbedSparseService,
    QdrantVectorStore,
    RAGService
)
from src.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management.
    
    All services are initialized ONCE at startup and shared across all requests.
    """
    # Setup logging
    setup_logging(level=settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting application...")
    
    # Initialize all services ONCE
    logger.info("Initializing services...")
    
    # LLM Service
    llm_service = GeminiLLMService(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    
    # Embedding Services
    dense_embedder = GeminiEmbeddingService(
        api_key=settings.embedding_api_key,
        model=settings.embedding_model
    )
    sparse_embedder = FastEmbedSparseService()
    
    # Qdrant Client
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    logger.info(f"Connected to Qdrant at {settings.qdrant_url}")
    
    # Vector Store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        collection_name=settings.qdrant_collection_name
    )
    
    # Prompt Manager
    prompt_manager = PromptManager()
    
    # RAG Service (Business Logic Layer)
    rag_service = RAGService(
        llm=llm_service,
        vector_store=vector_store,
        prompt_manager=prompt_manager,
        rag_k=settings.rag_k
    )
    
    # Store in app.state for access in endpoints
    app.state.settings = settings
    app.state.llm_service = llm_service
    app.state.vector_store = vector_store
    app.state.rag_service = rag_service
    
    logger.info("All services initialized successfully!")
    
    yield  # Application is running
    
    # Shutdown cleanup
    logger.info("Shutting down application...")
    qdrant_client.close()
    logger.info("Cleanup complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="NTT DATA RAG API",
    description="Hybrid Search RAG with LangGraph, Gemini, and Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NTT DATA RAG API - Hybrid Search with LangGraph",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "ask": "/api/v1/ask"
    }
