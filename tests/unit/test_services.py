"""Unit tests for services (ASYNC)."""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock

from src.services.rag_service import RAGService, RAGResponse
from src.core.interfaces import BaseLLMService, BaseVectorStore


class TestRAGService:
    """Unit tests for RAGService."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM that returns structured response."""
        llm = Mock(spec=BaseLLMService)
        # Rewrite node expects YEARS:/QUERY: format
        llm.generate.side_effect = [
            "YEARS: [2023]\nQUERY: What is sustainability?",  # rewrite
            "This is the final answer about sustainability."   # generate
        ]
        return llm
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        vs = Mock(spec=BaseVectorStore)
        vs.search.return_value = [
            "Document 1: NTT DATA sustainability report.",
            "Document 2: Carbon neutrality goals."
        ]
        vs.advanced_search.return_value = [
            "Document 1: NTT DATA sustainability report.",
            "Document 2: Carbon neutrality goals."
        ]
        return vs
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        pm = Mock()
        pm.get.return_value = "{question}"
        pm.get_system.return_value = ""
        return pm
    
    def test_rag_service_initialization(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test that RAGService initializes correctly."""
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager,
            rag_k=5
        )
        
        assert service.llm == mock_llm
        assert service.vector_store == mock_vector_store
        assert service.rag_k == 5
    
    @pytest.mark.asyncio
    async def test_ask_returns_rag_response(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test that ask() returns a proper RAGResponse."""
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        response = await service.ask("What is sustainability?")
        
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert isinstance(response.sources, list)
    
    @pytest.mark.asyncio
    async def test_ask_calls_vector_store_advanced_search(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test that ask() calls vector store advanced_search for hybrid retrieval."""
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager,
            rag_k=3
        )
        
        await service.ask("What is carbon neutrality?")
        
        # Vector store advanced_search should be called
        mock_vector_store.advanced_search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ask_calls_llm_generate_for_answer(self, mock_llm, mock_vector_store, mock_prompt_manager):
        """Test that ask() calls LLM generate for answer generation.
        
        Note: Rewrite now uses structured output (get_structured_llm), 
        so generate is only called once for the final answer.
        """
        service = RAGService(
            llm=mock_llm,
            vector_store=mock_vector_store,
            prompt_manager=mock_prompt_manager
        )
        
        await service.ask("2023 sustainability report")
        
        # LLM generate should be called once (for final answer)
        # Rewrite uses structured output separately
        assert mock_llm.generate.call_count == 1


class TestRAGResponse:
    """Unit tests for RAGResponse dataclass."""
    
    def test_rag_response_creation(self):
        """Test RAGResponse can be created with all fields."""
        response = RAGResponse(
            answer="Test answer",
            sources=["source1.pdf", "source2.pdf"],
            rewritten_question="Rewritten query",
            years_extracted=[2021, 2022, 2023]
        )
        
        assert response.answer == "Test answer"
        assert len(response.sources) == 2
        assert response.years_extracted == [2021, 2022, 2023]
    
    def test_rag_response_optional_fields(self):
        """Test RAGResponse works with optional fields as None."""
        response = RAGResponse(
            answer="Test answer",
            sources=[]
        )
        
        assert response.rewritten_question is None
        assert response.years_extracted is None
