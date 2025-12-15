"""Unit tests for core components."""
import pytest
from src.core.state import GraphState
from src.core.exceptions import RAGException, LLMException, VectorStoreException, PromptException


class TestGraphState:
    """Unit tests for GraphState TypedDict."""
    
    def test_graph_state_structure(self):
        """Test GraphState TypedDict structure."""
        state: GraphState = {
            "question": "What is sustainability?",
            "rewritten_question": "NTT DATA sustainability strategy",
            "documents": ["doc1", "doc2"],
            "answer": "Sustainability is...",
            "years": [2021, 2022]
        }
        
        assert state["question"] == "What is sustainability?"
        assert len(state["documents"]) == 2
        assert state["years"] == [2021, 2022]
    
    def test_graph_state_optional_years(self):
        """Test GraphState with no years."""
        state: GraphState = {
            "question": "General question",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        assert state["years"] is None


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
