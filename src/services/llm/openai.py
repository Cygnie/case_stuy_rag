"""OpenAI LLM service implementation."""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type

from src.core.interfaces import BaseLLMService
from src.core.exceptions import LLMException

logger = logging.getLogger(__name__)


class OpenAILLMService(BaseLLMService):
    """LLM service using OpenAI GPT models."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini", 
        temperature: float = 0.7
    ):
        """Initialize OpenAI LLM service.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
            temperature: Temperature for generation (0.0-1.0)
            
        Raises:
            LLMException: If initialization fails
        """
        try:
            self.api_key = api_key
            self.model = model
            self.temperature = temperature
            self.llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature,
                timeout=30,
                max_retries=2
            )
            logger.info(f"OpenAI LLM initialized: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise LLMException(f"Failed to initialize OpenAI LLM: {e}") from e
    
    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_delay(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True
    )
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text from prompt with automatic retry.
        
        Args:
            prompt: User prompt
            system: System message
            
        Returns:
            Generated text
            
        Raises:
            LLMException: If generation fails after retries
        """
        logger.debug(f"OpenAI generate called with prompt length: {len(prompt)}")
        
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages)
            logger.debug(f"OpenAI response generated: {len(response.content)} chars")
            return response.content
        except (TimeoutError, ConnectionError):
            # Let tenacity handle retries
            raise
        except Exception as e:
            logger.error(f"OpenAI generation failed: {type(e).__name__}: {e}")
            raise LLMException(f"Failed to generate response: {e}") from e
    
    def get_structured_llm(self, schema):
        """Get LLM configured for structured output with given Pydantic schema.
        
        Args:
            schema: Pydantic BaseModel class for structured output
            
        Returns:
            LLM instance configured for structured output
        """
        return self.llm.with_structured_output(schema)
