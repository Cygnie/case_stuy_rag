"""Dependency providers for dependency injection."""
from fastapi import Request, HTTPException

from src.core.config import settings, Settings
from src.services.rag_service import RAGService


def get_settings() -> Settings:
    """Provide application settings.
    
    Returns:
        Application settings instance
    """
    return settings


def get_rag_service(request: Request) -> RAGService:
    """Provide RAG service for dependency injection.
    
    Args:
        request: FastAPI request object
        
    Returns:
        RAG service instance
        
    Raises:
        HTTPException: If service not available
    """
    # Check container exists
    if not hasattr(request.app.state, 'container'):
        raise HTTPException(
            status_code=503,
            detail="Service container not initialized"
        )
    
    # Get RAG service from container
    rag = request.app.state.container.rag_service
    if not rag:
        raise HTTPException(
            status_code=503,
            detail="RAG service not available"
        )
    
    return rag
