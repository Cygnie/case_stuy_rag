"""Service container for managing application-wide singleton services.

This module provides a container class that holds all initialized services
and manages their lifecycle (initialization and cleanup).
"""
import logging
from typing import Optional

from src.core.interfaces import BaseLLMService, BaseEmbeddingService, BaseVectorStore
from src.services.rag_service import RAGService
from src.prompts.prompts import PromptManager

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Container for application-wide singleton services.
    
    This container holds all initialized services and provides:
    - Centralized service storage
    - Lifecycle management (startup/shutdown)
    - Easy access to services throughout the application
    
    Attributes:
        llm_service: Language model service
        dense_embedder: Dense embedding service
        sparse_embedder: Sparse embedding service
        vector_store: Vector database service
        prompt_manager: Prompt template manager
        rag_service: Main RAG service (business logic)
    """
    
    def __init__(self):
        """Initialize empty container."""
        self.llm_service: Optional[BaseLLMService] = None
        self.dense_embedder: Optional[BaseEmbeddingService] = None
        self.sparse_embedder: Optional[BaseEmbeddingService] = None
        self.vector_store: Optional[BaseVectorStore] = None
        self.prompt_manager: Optional[PromptManager] = None
        self.rag_service: Optional[RAGService] = None
        
        # For cleanup
        self._qdrant_client = None
    
    async def shutdown(self) -> None:
        """Clean up resources on application shutdown.
        
        Closes connections and releases resources held by services.
        """
        logger.info("Shutting down service container...")
        
        # Close Qdrant connection if exists
        if self._qdrant_client:
            try:
                self._qdrant_client.close()
                logger.info("Qdrant client closed")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}")
        
        logger.info("Service container shutdown complete")
    
    def __repr__(self) -> str:
        """String representation showing which services are initialized."""
        services = {
            "llm": self.llm_service is not None,
            "dense_embedder": self.dense_embedder is not None,
            "sparse_embedder": self.sparse_embedder is not None,
            "vector_store": self.vector_store is not None,
            "prompt_manager": self.prompt_manager is not None,
            "rag_service": self.rag_service is not None,
        }
        initialized = [name for name, ready in services.items() if ready]
        return f"<ServiceContainer: {', '.join(initialized)}>"
