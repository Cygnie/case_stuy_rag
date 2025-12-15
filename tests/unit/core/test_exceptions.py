"""Unit tests for custom exceptions."""
import pytest
from src.core.exceptions import RAGException, LLMException, VectorStoreException, PromptException


class TestExceptions:
    """Unit tests for custom exceptions."""
    
    def test_rag_exception_hierarchy(self):
        """Test exception inheritance."""
        assert issubclass(LLMException, RAGException)
        assert issubclass(VectorStoreException, RAGException)
        assert issubclass(PromptException, RAGException)
    
    def test_llm_exception_message(self):
        """Test LLMException can carry message."""
        exc = LLMException("API key invalid")
        assert str(exc) == "API key invalid"
    
    def test_vector_store_exception_message(self):
        """Test VectorStoreException message."""
        exc = VectorStoreException("Connection refused")
        assert "Connection refused" in str(exc)
    
    def test_exception_can_be_raised_and_caught(self):
        """Test exception can be raised and caught by parent."""
        with pytest.raises(RAGException):
            raise LLMException("Test error")
    
    def test_llm_exception_inherits_from_rag_exception(self):
        """Test exception hierarchy."""
        exc = LLMException("test error")
        assert isinstance(exc, RAGException)
    
    def test_exception_can_be_caught_by_parent(self):
        """Test catching child exceptions with parent type."""
        def raise_llm_exception():
            raise LLMException("API error")
        
        with pytest.raises(RAGException):
            raise_llm_exception()
    
    def test_exception_chaining(self):
        """Test exception chaining with 'from'."""
        original = ValueError("original error")
        chained = VectorStoreException("wrapped error")
        chained.__cause__ = original
        
        assert chained.__cause__ is original
