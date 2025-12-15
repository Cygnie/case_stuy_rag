"""Custom exceptions for the RAG application."""

class RAGException(Exception):
    """Base exception for RAG application."""
    pass


class LLMException(RAGException):
    """Exception raised when LLM operations fail."""
    pass


class EmbeddingException(RAGException):
    """Exception raised when embedding operations fail."""
    pass


class VectorStoreException(RAGException):
    """Exception raised when vector store operations fail."""
    pass


class ConfigurationException(RAGException):
    """Exception raised when configuration is invalid."""
    pass


class PromptException(RAGException):
    """Exception raised when prompt loading/formatting fails."""
    pass
