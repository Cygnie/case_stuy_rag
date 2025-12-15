"""FastEmbed sparse embedding service."""
import logging
from fastembed import SparseTextEmbedding

from src.core.interfaces import BaseEmbeddingService
from src.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class FastEmbedSparseService(BaseEmbeddingService):
    """Sparse vector embeddings using FastEmbed BM25."""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """Initialize FastEmbed sparse service.
        
        Args:
            model_name: Sparse embedding model name
            
        Raises:
            EmbeddingException: If initialization fails
        """
        try:
            self.model_name = model_name
            self.model = SparseTextEmbedding(model_name=model_name)
            logger.info(f"FastEmbed sparse embeddings initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed: {e}")
            raise EmbeddingException(f"Failed to initialize FastEmbed: {e}") from e
    
    def embed(self, text: str) -> dict:
        """Generate sparse embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Sparse vector dictionary
            
        Raises:
            EmbeddingException: If embedding fails
        """
        logger.debug(f"Generating sparse embedding for text length: {len(text)}")
        try:
            embeddings = list(self.model.embed([text]))
            return embeddings[0]
        except Exception as e:
            logger.error(f"Sparse embedding failed: {type(e).__name__}: {e}")
            raise EmbeddingException(f"Failed to generate sparse embedding: {e}") from e
