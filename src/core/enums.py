"""Enums for domain types."""
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    FASTEMBED_SPARSE = "fastembed_sparse"
    MOCK = "mock"  # For testing


class VectorStoreProvider(str, Enum):
    """Supported vector store providers."""
    QDRANT = "qdrant"
    CHROMA = "chroma"
    MOCK = "mock"  # For testing
