"""Factory for creating vector store service instances."""
import logging
from src.core.interfaces import BaseVectorStore
from src.core.enums import VectorStoreProvider
from src.core.exceptions import VectorStoreException

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Simple factory for creating vector store service instances."""
    
    @classmethod
    def create(
        cls,
        provider: VectorStoreProvider,
        **kwargs
    ) -> BaseVectorStore:
        """Create vector store service instance.
        
        Args:
            provider: Vector store provider to use
            **kwargs: Provider-specific parameters (client, embedders, etc.)
            
        Returns:
            Configured vector store instance
            
        Raises:
            VectorStoreException: If provider unknown or creation fails
        """
        # Import here to avoid circular imports
        from src.services.vector_stores.qdrant import QdrantVectorStore
        
        try:
            if provider == VectorStoreProvider.QDRANT:
                logger.info("Creating Qdrant vector store")
                return QdrantVectorStore(**kwargs)
            else:
                raise VectorStoreException(f"Unknown vector store provider: {provider}")
        except Exception as e:
            if isinstance(e, VectorStoreException):
                raise
            logger.error(f"Failed to create {provider} vector store: {e}")
            raise VectorStoreException(f"Failed to create {provider} vector store: {e}") from e
