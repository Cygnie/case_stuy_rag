"""Embedding services for dense and sparse vectors with retry logic."""
import logging
import google.generativeai as genai
from fastembed import SparseTextEmbedding
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from src.core.interfaces import BaseEmbeddingService
from src.core.exceptions import EmbeddingException

logger = logging.getLogger(__name__)


class GeminiEmbeddingService(BaseEmbeddingService):
    """Dense vector embeddings using Google Gemini."""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        """Initialize Gemini embedding service.
        
        Args:
            api_key: Google API key
            model: Embedding model name
            
        Raises:
            EmbeddingException: If initialization fails
        """
        try:
            self.api_key = api_key
            self.model = model
            genai.configure(api_key=api_key)
            logger.info(f"Gemini embeddings initialized: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise EmbeddingException(f"Failed to initialize Gemini embeddings: {e}") from e
    
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, TimeoutError, ConnectionError)),
        stop=stop_after_delay(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True
    )
    def embed(self, text: str) -> list[float]:
        """Generate dense embedding for text with retry (max 5 seconds).
        
        Args:
            text: Text to embed
            
        Returns:
            Dense embedding vector (768 dimensions)
            
        Raises:
            EmbeddingException: If embedding fails after retries
        """
        logger.debug(f"Generating dense embedding for text length: {len(text)}")
        try:
            result = genai.embed_content(model=self.model, content=text)
            return result['embedding']
        except (ResourceExhausted, ServiceUnavailable, TimeoutError, ConnectionError):
            raise
        except Exception as e:
            logger.error(f"Embedding generation failed: {type(e).__name__}: {e}")
            raise EmbeddingException(f"Failed to generate embedding: {e}") from e


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
