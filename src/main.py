"""FastAPI application with RAG workflow using LangGraph."""
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.container import build_container
from src.api.v1.endpoints import health, rag
from src.core.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management.
    
    Initializes all services once at startup and stores them in a container.
    Cleans up resources on shutdown.
    """
    # Setup logging
    setup_logging(level=settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting application...")
    
    # Build service container
    try:
        container = build_container(settings)
    except Exception as e:
        logger.critical(f"Failed to build service container: {e}")
        logger.critical("Application cannot start. Exiting.")
        sys.exit(1)
    
    # Store container in app state
    app.state.container = container
    logger.info("Application started successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    await container.shutdown()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
