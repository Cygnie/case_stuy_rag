"""Unit tests for GraphState TypedDict."""
import pytest
from src.core.state import GraphState


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
    
    def test_graph_state_with_empty_documents(self):
        """Test GraphState with empty documents list."""
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
        state: GraphState = {
            "question": "test",
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        }
        
        assert len(state["years"]) == 7
