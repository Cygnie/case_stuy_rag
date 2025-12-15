"""Abstract interfaces for services."""
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed(self, text: str) -> Any:
        """Generate embedding for text."""
        pass


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_structured_llm(self, schema):
        """Get LLM configured for structured output with given Pydantic schema."""
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector store services."""
    
    @abstractmethod
    def add_documents(self, docs: list[dict]) -> None:
        """Add documents to vector store."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 4) -> list[str]:
        """Simple search using only dense embeddings (cosine similarity)."""
        pass
    
    @abstractmethod
    def advanced_search(self, query: str, years: Optional[list[int]] = None, k: int = 4) -> list[str]:
        """Advanced hybrid search using dense + sparse embeddings with RRF fusion."""
        pass
