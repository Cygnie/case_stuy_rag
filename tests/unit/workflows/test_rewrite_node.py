"""Unit tests for RewriteNode."""
import pytest
from unittest.mock import Mock, MagicMock
from pydantic import BaseModel

from src.workflows.nodes.rewrite import RewriteNode, RewriteOutput
from src.core.state import GraphState
from src.core.interfaces import BaseLLMService
from src.core.prompts import PromptManager


class TestRewriteNode:
    """Unit tests for RewriteNode."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        llm = Mock(spec=BaseLLMService)
        
        # Mock structured LLM
        structured_llm = Mock()
        structured_llm.invoke = Mock(return_value=RewriteOutput(
            years=[2023],
            query="What is sustainability in 2023?"
        ))
        llm.get_structured_llm = Mock(return_value=structured_llm)
        
        return llm
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        pm = Mock(spec=PromptManager)
        pm.get_system.return_value = "You are a helpful assistant"
        pm.get.return_value = "Rewrite this question: {question}"
        return pm
    
    def test_rewrite_node_initialization(self, mock_llm, mock_prompt_manager):
        """Test RewriteNode initialization."""
        node = RewriteNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        assert node.llm == mock_llm
        assert node.prompt_manager == mock_prompt_manager
        assert node.structured_llm is not None
    
    def test_rewrite_node_execution(self, mock_llm, mock_prompt_manager):
        """Test RewriteNode execution with year extraction."""
        node = RewriteNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "2023 sustainability report",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert result["rewritten_question"] == "What is sustainability in 2023?"
        assert result["years"] == [2023]
        mock_llm.get_structured_llm.assert_called_once()
    
    def test_rewrite_node_with_no_years(self, mock_llm, mock_prompt_manager):
        """Test RewriteNode when no years are detected."""
        # Mock structured output with no years
        structured_llm = Mock()
        structured_llm.invoke = Mock(return_value=RewriteOutput(
            years=[],
            query="general sustainability question"
        ))
        mock_llm.get_structured_llm = Mock(return_value=structured_llm)
        
        node = RewriteNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "What is sustainability?",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert result["rewritten_question"] == "general sustainability question"
        assert result["years"] is None
    
    def test_rewrite_node_with_multiple_years(self, mock_llm, mock_prompt_manager):
        """Test RewriteNode with year range extraction."""
        structured_llm = Mock()
        structured_llm.invoke = Mock(return_value=RewriteOutput(
            years=[2021, 2022, 2023],
            query="carbon footprint 2021-2023"
        ))
        mock_llm.get_structured_llm = Mock(return_value=structured_llm)
        
        node = RewriteNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "2021-2023 carbon footprint",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert result["years"] == [2021, 2022, 2023]
    
    def test_rewrite_node_fallback_on_error(self, mock_llm, mock_prompt_manager):
        """Test RewriteNode fallback behavior when structured output fails."""
        # Mock structured LLM to raise exception
        structured_llm = Mock()
        structured_llm.invoke = Mock(side_effect=Exception("API Error"))
        mock_llm.get_structured_llm = Mock(return_value=structured_llm)
        
        node = RewriteNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "test question",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        # Should fallback to original question
        assert result["rewritten_question"] == "test question"
        assert result["years"] is None
