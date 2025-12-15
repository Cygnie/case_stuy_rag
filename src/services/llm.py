"""LLM service using Google Gemini with retry logic."""
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from src.core.interfaces import BaseLLMService
from src.core.exceptions import LLMException

logger = logging.getLogger(__name__)


class GeminiLLMService(BaseLLMService):
    """LLM service using Google Gemini via LangChain."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.7):
        """Initialize Gemini LLM service.
        
        Args:
            api_key: Google API key
            model: Gemini model name
            temperature: Temperature for generation (0.0-1.0)
            
        Raises:
            LLMException: If initialization fails
        """
        try:
            self.api_key = api_key
            self.model = model
            self.temperature = temperature
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature
            )
            logger.info(f"Gemini LLM initialized: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise LLMException(f"Failed to initialize Gemini LLM: {e}") from e
    
    @retry(
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, TimeoutError, ConnectionError)),
        stop=stop_after_delay(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True
    )
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text from prompt with automatic retry (max 5 seconds).
        
        Args:
            prompt: User prompt
            system: System message
            
        Returns:
            Generated text
            
        Raises:
            LLMException: If generation fails after retries
        """
        logger.debug(f"LLM generate called with prompt length: {len(prompt)}")
        
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        
        try:
            response = self.llm.invoke(messages)
            logger.debug(f"LLM response generated: {len(response.content)} chars")
            return response.content
        except (ResourceExhausted, ServiceUnavailable, TimeoutError, ConnectionError):
            # Let tenacity handle retries
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {type(e).__name__}: {e}")
            raise LLMException(f"Failed to generate response: {e}") from e
    
    def get_structured_llm(self, schema):
        """Get LLM configured for structured output with given Pydantic schema.
        
        Args:
            schema: Pydantic BaseModel class for structured output
            
        Returns:
            LLM instance configured for structured output
        """
        return self.llm.with_structured_output(schema)
