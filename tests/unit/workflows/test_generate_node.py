"""Unit tests for GenerateNode."""
import pytest
from unittest.mock import Mock

from src.workflows.nodes.generate import GenerateNode
from src.core.state import GraphState
from src.core.interfaces import BaseLLMService
from src.core.prompts import PromptManager


class TestGenerateNode:
    """Unit tests for GenerateNode."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        llm = Mock(spec=BaseLLMService)
        llm.generate = Mock(return_value="This is the generated answer about sustainability.")
        return llm
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager."""
        pm = Mock(spec=PromptManager)
        pm.get_system.return_value = "You are a helpful assistant"
        pm.get.return_value = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return pm
    
    def test_generate_node_initialization(self, mock_llm, mock_prompt_manager):
        """Test GenerateNode initialization."""
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        assert node.llm == mock_llm
        assert node.prompt_manager == mock_prompt_manager
    
    def test_generate_node_execution(self, mock_llm, mock_prompt_manager):
        """Test GenerateNode generates answer from context."""
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "What is sustainability?",
            "rewritten_question": "sustainability strategy",
            "documents": [
                "Doc 1: NTT DATA focuses on sustainability",
                "Doc 2: Carbon neutrality by 2040"
            ],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert len(result["answer"]) > 0
        assert result["answer"] == "This is the generated answer about sustainability."
        mock_llm.generate.assert_called_once()
    
    def test_generate_node_formats_context(self, mock_llm, mock_prompt_manager):
        """Test that GenerateNode formats context correctly."""
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "test question",
            "rewritten_question": "test",
            "documents": ["Doc 1", "Doc 2", "Doc 3"],
            "answer": "",
            "years": None
        }
        
        node.execute(state)
        
        # LLM generate should be called
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        
        # Prompt should contain context
        prompt = call_args[0][0]
        assert "Doc 1" in prompt or "Context" in prompt
    
    def test_generate_node_with_empty_documents(self, mock_llm, mock_prompt_manager):
        """Test GenerateNode with empty documents list."""
        mock_llm.generate = Mock(return_value="I don't have enough information.")
        
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "test question",
            "rewritten_question": "test",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        # Should still generate an answer (even if context is empty)
        assert len(result["answer"]) > 0
        mock_llm.generate.assert_called_once()
    
    def test_generate_node_uses_original_question(self, mock_llm, mock_prompt_manager):
        """Test that GenerateNode uses original question in prompt."""
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "What is NTT DATA?",
            "rewritten_question": "optimized query",
            "documents": ["Doc 1"],
            "answer": "",
            "years": None
        }
        
        node.execute(state)
        
        # Should use original question for generating answer
        call_args = mock_llm.generate.call_args
        prompt = call_args[0][0]
        assert "What is NTT DATA?" in prompt
    
    def test_generate_node_strips_whitespace(self, mock_llm, mock_prompt_manager):
        """Test that GenerateNode strips whitespace from answer."""
        mock_llm.generate = Mock(return_value="  Answer with whitespace  \n")
        
        node = GenerateNode(llm=mock_llm, prompt_manager=mock_prompt_manager)
        
        state: GraphState = {
            "question": "test",
            "rewritten_question": "test",
            "documents": ["doc"],
            "answer": "",
            "years": None
        }
        
        result = node.execute(state)
        
        assert result["answer"] == "Answer with whitespace"
