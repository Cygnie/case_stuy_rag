"""Unit tests for RetrieveNode."""
import pytest
from unittest.mock import Mock

from src.workflows.nodes.retrieve import RetrieveNode
from src.core.state import GraphState
from src.core.interfaces import BaseVectorStore


class TestRetrieveNode:
    """Unit tests for RetrieveNode."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        vs = Mock(spec=BaseVectorStore)
        vs.advanced_search = Mock(return_value=[
            "Document 1: NTT DATA sustainability",
            "Document 2: Carbon neutrality goals"
        ])
        return vs
    
    def test_retrieve_node_initialization(self, mock_vector_store):
        """Test RetrieveNode initialization."""
        node = RetrieveNode(vector_store=mock_vector_store, k=5)
        
        assert node.vector_store == mock_vector_store
        assert node.k == 5
    
    def test_retrieve_node_execution(self, mock_vector_store):
        """Test RetrieveNode retrieves documents."""
        node = RetrieveNode(vector_store=mock_vector_store, k=5)
        
        state: GraphState = {
            "question": "What is sustainability?",
            "rewritten_question": "sustainability strategy",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert len(result["documents"]) == 2
        assert "NTT DATA sustainability" in result["documents"][0]
        mock_vector_store.advanced_search.assert_called_once()
    
    def test_retrieve_node_uses_rewritten_question(self, mock_vector_store):
        """Test that RetrieveNode uses rewritten question for search."""
        node = RetrieveNode(vector_store=mock_vector_store, k=3)
        
        state: GraphState = {
            "question": "original question",
            "rewritten_question": "optimized query",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        node.execute(state)
        
        # Should use rewritten question
        call_args = mock_vector_store.advanced_search.call_args
        assert call_args.kwargs["query"] == "optimized query"
    
    def test_retrieve_node_with_years_filter(self, mock_vector_store):
        """Test RetrieveNode passes year filter to vector store."""
        node = RetrieveNode(vector_store=mock_vector_store, k=5)
        
        state: GraphState = {
            "question": "2023 report",
            "rewritten_question": "2023 sustainability report",
            "documents": [],
            "answer": "",
            "years": [2023]
        }
        
        node.execute(state)
        
        # Should pass years to advanced_search
        call_args = mock_vector_store.advanced_search.call_args
        assert call_args.kwargs["years"] == [2023]
    
    def test_retrieve_node_with_custom_k(self, mock_vector_store):
        """Test RetrieveNode with custom k value."""
        node = RetrieveNode(vector_store=mock_vector_store, k=10)
        
        state: GraphState = {
            "question": "test",
            "rewritten_question": "test",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        node.execute(state)
        
        # Should use custom k
        call_args = mock_vector_store.advanced_search.call_args
        assert call_args.kwargs["k"] == 10
