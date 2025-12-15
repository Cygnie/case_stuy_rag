"""Factory for creating LLM service instances."""
import logging
from src.core.interfaces import BaseLLMService
from src.core.enums import LLMProvider
from src.core.exceptions import LLMException

logger = logging.getLogger(__name__)


class LLMFactory:
    """Simple factory for creating LLM service instances."""
    
    @classmethod
    def create(
        cls,
        provider: LLMProvider,
        **kwargs
    ) -> BaseLLMService:
        """Create LLM service instance.
        
        Args:
            provider: LLM provider to use (GEMINI or OPENAI)
            **kwargs: Provider-specific parameters (api_key, model, etc.)
            
        Returns:
            Configured LLM service instance
            
        Raises:
            LLMException: If provider unknown or creation fails
        """
        # Import here to avoid circular imports
        from src.services.llm.gemini import GeminiLLMService
        from src.services.llm.openai import OpenAILLMService
        
        try:
            if provider == LLMProvider.GEMINI:
                logger.info(f"Creating Gemini LLM service")
                return GeminiLLMService(**kwargs)
            elif provider == LLMProvider.OPENAI:
                logger.info(f"Creating OpenAI LLM service")
                return OpenAILLMService(**kwargs)
            else:
                raise LLMException(f"Unknown LLM provider: {provider}")
        except Exception as e:
            if isinstance(e, LLMException):
                raise
            logger.error(f"Failed to create {provider} LLM service: {e}")
            raise LLMException(f"Failed to create {provider} LLM service: {e}") from e
