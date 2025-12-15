"""Unit tests for RAGGraph."""
import pytest
from unittest.mock import Mock, AsyncMock

from src.workflows.graph import RAGGraph
from src.core.state import GraphState


class TestRAGGraph:
    """Unit tests for RAGGraph compilation and execution."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="Mock answer")
        mock_llm.get_structured_llm = Mock()
        
        mock_vector_store = Mock()
        mock_vector_store.advanced_search = Mock(return_value=["doc1", "doc2"])
        
        mock_prompt_manager = Mock()
        mock_prompt_manager.get = Mock(return_value=Mock(format=Mock(return_value="formatted prompt")))
        mock_prompt_manager.get_system = Mock(return_value="system prompt")
        
        return mock_llm, mock_vector_store, mock_prompt_manager
    
    def test_graph_initialization(self, mock_services):
        """Test that RAGGraph initializes with services."""
        llm, vector_store, prompt_manager = mock_services
        
        graph = RAGGraph(
            llm=llm,
            vector_store=vector_store,
            prompt_manager=prompt_manager,
            rag_k=5
        )
        
        assert graph.llm == llm
        assert graph.vector_store == vector_store
        assert graph.prompt_manager == prompt_manager
        assert graph.rag_k == 5
        assert graph._compiled is None  # Not compiled yet
        assert graph.rewrite_node is not None
        assert graph.retrieve_node is not None
        assert graph.generate_node is not None
    
    def test_graph_build_caching(self, mock_services):
        """Test that graph.build() caches compiled graph."""
        llm, vector_store, prompt_manager = mock_services
        
        graph = RAGGraph(
            llm=llm,
            vector_store=vector_store,
            prompt_manager=prompt_manager
        )
        
        # First build - should compile
        compiled1 = graph.build()
        assert compiled1 is not None
        assert graph._compiled is not None
        
        # Second build - should return cached
        compiled2 = graph.build()
        assert compiled2 is compiled1  # Same object
    
    def test_graph_compilation(self, mock_services):
        """Test that graph compiles successfully."""
        llm, vector_store, prompt_manager = mock_services
        
        graph = RAGGraph(
            llm=llm,
            vector_store=vector_store,
            prompt_manager=prompt_manager
        )
        
        compiled = graph.build()
        
        # Compiled graph should be a runnable
        assert compiled is not None
        assert hasattr(compiled, 'invoke')
        assert hasattr(compiled, 'ainvoke')
    
    @pytest.mark.asyncio
    async def test_graph_run_helper(self, mock_services):
        """Test that graph.run() helper works."""
        llm, vector_store, prompt_manager = mock_services
        
        graph = RAGGraph(
            llm=llm,
            vector_store=vector_store,
            prompt_manager=prompt_manager
        )
        
        # Test run() method
        result = await graph.run("What is NTT DATA?")
        
        # Should have all state keys
        assert "question" in result
        assert "rewritten_question" in result
        assert "documents" in result
        assert "answer" in result
        assert "years" in result
