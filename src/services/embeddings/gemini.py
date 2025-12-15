"""Gemini embedding service for dense vectors with retry logic."""
import logging
import google.generativeai as genai
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
