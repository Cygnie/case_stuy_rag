"""Factory for creating embedding service instances."""
import logging
from src.core.interfaces import BaseEmbeddingService
from src.core.enums import EmbeddingProvider
from src.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Simple factory for creating embedding service instances."""
    
    @classmethod
    def create(
        cls,
        provider: EmbeddingProvider,
        **kwargs
    ) -> BaseEmbeddingService:
        """Create embedding service instance.
        
        Args:
            provider: Embedding provider to use
            **kwargs: Provider-specific parameters (api_key, model, etc.)
            
        Returns:
            Configured embedding service instance
            
        Raises:
            EmbeddingException: If provider unknown or creation fails
        """
        # Import here to avoid circular imports
        from src.services.embeddings.gemini import GeminiEmbeddingService
        from src.services.embeddings.fastembed import FastEmbedSparseService
        
        try:
            if provider == EmbeddingProvider.GEMINI:
                logger.info("Creating Gemini embedding service")
                return GeminiEmbeddingService(**kwargs)
            elif provider == EmbeddingProvider.FASTEMBED_SPARSE:
                logger.info("Creating FastEmbed sparse service")
                return FastEmbedSparseService(**kwargs)
            else:
                raise EmbeddingException(f"Unknown embedding provider: {provider}")
        except Exception as e:
            if isinstance(e, EmbeddingException):
                raise
            logger.error(f"Failed to create {provider} embedding service: {e}")
            raise EmbeddingException(f"Failed to create {provider} embedding service: {e}") from e
