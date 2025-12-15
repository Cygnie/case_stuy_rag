"""Unit tests for PromptManager."""
import pytest
from src.prompts.prompts import PromptManager
from src.core.exceptions import PromptException


class TestPromptManager:
    """Unit tests for PromptManager."""
    
    def test_prompt_manager_with_missing_key(self):
        """Test handling of missing prompt key."""
        pm = PromptManager()
        
        with pytest.raises(PromptException):
            pm.get("nonexistent_node", "nonexistent_key")
    
    def test_prompt_manager_get_valid_key(self):
        """Test getting valid prompt."""
        pm = PromptManager()
        
        # Should not raise for valid keys
        template = pm.get("rewrite", "template")
        assert len(template) > 0
    
    def test_prompt_manager_get_system_prompt(self):
        """Test getting system prompt."""
        pm = PromptManager()
        
        system_prompt = pm.get_system("rewrite")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
