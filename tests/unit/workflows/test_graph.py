"""Unit tests for RAGGraph."""
import pytest
from unittest.mock import Mock

from src.workflows.graph import RAGGraph
from src.workflows.nodes import RewriteNode, RetrieveNode, GenerateNode
from src.core.state import GraphState


class TestRAGGraph:
    """Unit tests for RAGGraph compilation and execution."""
    
    @pytest.fixture
    def mock_nodes(self):
        """Create mock nodes for testing."""
        rewrite = Mock(spec=RewriteNode)
        rewrite.execute = Mock(return_value={
            "question": "test",
            "rewritten_question": "test query",
            "documents": [],
            "answer": "",
            "years": None
        })
        
        retrieve = Mock(spec=RetrieveNode)
        retrieve.execute = Mock(return_value={
            "question": "test",
            "rewritten_question": "test query",
            "documents": ["doc1", "doc2"],
            "answer": "",
            "years": None
        })
        
        generate = Mock(spec=GenerateNode)
        generate.execute = Mock(return_value={
            "question": "test",
            "rewritten_question": "test query",
            "documents": ["doc1", "doc2"],
            "answer": "This is the answer",
            "years": None
        })
        
        return rewrite, retrieve, generate
    
    def test_graph_initialization(self, mock_nodes):
        """Test that RAGGraph initializes with nodes."""
        rewrite, retrieve, generate = mock_nodes
        
        graph = RAGGraph(
            rewrite_node=rewrite,
            retrieve_node=retrieve,
            generate_node=generate
        )
        
        assert graph.rewrite_node == rewrite
        assert graph.retrieve_node == retrieve
        assert graph.generate_node == generate
    
    def test_graph_compilation(self, mock_nodes):
        """Test that graph compiles successfully."""
        rewrite, retrieve, generate = mock_nodes
        
        graph = RAGGraph(
            rewrite_node=rewrite,
            retrieve_node=retrieve,
            generate_node=generate
        )
        
        compiled = graph.compile()
        
        # Compiled graph should be a runnable
        assert compiled is not None
        assert hasattr(compiled, 'invoke')
    
    def test_graph_execution_flow(self, mock_nodes):
        """Test that graph executes nodes in correct order."""
        rewrite, retrieve, generate = mock_nodes
        
        graph = RAGGraph(
            rewrite_node=rewrite,
            retrieve_node=retrieve,
            generate_node=generate
        )
        
        compiled = graph.compile()
        
        initial_state: GraphState = {
            "question": "What is NTT DATA?",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = compiled.invoke(initial_state)
        
        # All nodes should have been called
        rewrite.execute.assert_called_once()
        retrieve.execute.assert_called_once()
        generate.execute.assert_called_once()
        
        # Result should have answer
        assert result["answer"] == "This is the answer"
