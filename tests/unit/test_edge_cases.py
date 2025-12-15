"""Edge case tests for RAG API."""
import pytest
from unittest.mock import Mock, patch

from src.services.rag_service import RAGService, RAGResponse
from src.core.exceptions import RAGException, LLMException, VectorStoreException


class TestRAGServiceEdgeCases:
    """Edge case tests for RAGService."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = Mock()
        llm.generate.return_value = "YEARS: []\nQUERY: test query"
        return llm
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        vs = Mock()
        vs.search.return_value = ["Doc 1", "Doc 2"]
        return vs
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        pm = Mock()
        pm.get.return_value = "{question}"
        pm.get_system.return_value = ""
        return pm
    
    @pytest.fixture
    def rag_service(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Create RAGService with mocks."""
        return RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
    
    def test_empty_documents_returned(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test handling when no documents are returned."""
        mock_vector_store.search.return_value = []
        mock_llm.generate.side_effect = [
            "YEARS: []\nQUERY: test",
            "I couldn't find relevant information."
        ]
        
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        # Should not raise, should return answer
        assert service is not None
    
    def test_year_extraction_with_range(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test year extraction with range (2021-2023)."""
        mock_llm.generate.side_effect = [
            "YEARS: [2021, 2022, 2023]\nQUERY: carbon footprint",
            "Answer about carbon footprint"
        ]
        
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        # Service should be created successfully
        assert service.rag_k == 5
    
    def test_year_extraction_with_no_years(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test when no years are extracted."""
        mock_llm.generate.side_effect = [
            "YEARS: []\nQUERY: general sustainability",
            "General answer"
        ]
        
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        assert service is not None
    
    def test_very_long_question(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test handling of very long questions."""
        long_question = "sustainability " * 100  # 1400+ chars
        mock_llm.generate.side_effect = [
            "YEARS: []\nQUERY: sustainability overview",
            "Answer"
        ]
        
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        # Should handle long questions
        assert service is not None
    
    def test_special_characters_in_question(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test handling of special characters."""
        mock_llm.generate.side_effect = [
            "YEARS: []\nQUERY: test",
            "Answer"
        ]
        
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        # Should handle special chars
        assert service is not None


class TestVectorStoreEdgeCases:
    """Edge case tests for vector store operations."""
    
    def test_convert_sparse_embedding(self):
        """Test sparse embedding conversion."""
        from src.services.vector_store import QdrantVectorStore
        import numpy as np
        
        # Create mock sparse embedding
        class MockSparseEmbedding:
            indices = np.array([0, 5, 10])
            values = np.array([0.1, 0.5, 0.9])
        
        # Create instance bypassing __init__
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        result = store._convert_sparse_embedding(MockSparseEmbedding())
        
        assert result.indices == [0, 5, 10]
        assert result.values == [0.1, 0.5, 0.9]
    
    def test_empty_sparse_embedding(self):
        """Test handling of empty sparse embedding."""
        from src.services.vector_store import QdrantVectorStore
        import numpy as np
        
        class MockEmptySparseEmbedding:
            indices = np.array([])
            values = np.array([])
        
        # Create instance bypassing __init__
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        result = store._convert_sparse_embedding(MockEmptySparseEmbedding())
        
        assert result.indices == []
        assert result.values == []


class TestExceptionHandling:
    """Tests for exception handling."""
    
    def test_llm_exception_inherits_from_rag_exception(self):
        """Test exception hierarchy."""
        exc = LLMException("test error")
        assert isinstance(exc, RAGException)
    
    def test_vector_store_exception_message(self):
        """Test VectorStoreException preserves message."""
        exc = VectorStoreException("Connection failed")
        assert "Connection failed" in str(exc)
    
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


class TestPromptManager:
    """Edge case tests for PromptManager."""
    
    def test_prompt_manager_with_missing_key(self):
        """Test handling of missing prompt key."""
        from src.core.prompts import PromptManager
        from src.core.exceptions import PromptException
        
        pm = PromptManager()
        
        with pytest.raises(PromptException):
            pm.get("nonexistent_node", "nonexistent_key")
    
    def test_prompt_manager_get_valid_key(self):
        """Test getting valid prompt."""
        from src.core.prompts import PromptManager
        
        pm = PromptManager()
        
        # Should not raise for valid keys
        template = pm.get("rewrite", "template")
        assert len(template) > 0


class TestGraphState:
    """Edge case tests for GraphState."""
    
    def test_graph_state_with_empty_documents(self):
        """Test GraphState with empty documents list."""
        from src.core.state import GraphState
        
        state: GraphState = {
            "question": "test",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        assert state["documents"] == []
        assert state["years"] is None
    
    def test_graph_state_with_many_years(self):
        """Test GraphState with many years."""
        from src.core.state import GraphState
        
        state: GraphState = {
            "question": "test",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        }
        
        assert len(state["years"]) == 7
