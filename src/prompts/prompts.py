"""Prompt management from YAML configuration."""
import logging
from pathlib import Path
from typing import Optional
import yaml

from src.core.exceptions import PromptException

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages LLM prompts loaded from YAML file."""
    
    def __init__(self, prompts_path: Optional[str] = None):
        """Initialize PromptManager.
        
        Args:
            prompts_path: Path to prompts YAML file (default: src/prompts/prompts.yaml)
            
        Raises:
            PromptException: If loading prompts fails
        """
        if prompts_path is None:
            prompts_path = "src/prompts/prompts.yaml"
        self.prompts_path = Path(prompts_path)
        self._load_prompts()
        logger.info(f"PromptManager loaded from: {self.prompts_path}")
    
    def _load_prompts(self) -> None:
        """Load prompts from YAML file.
        
        Raises:
            PromptException: If file not found or invalid YAML
        """
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                self.prompts = yaml.safe_load(f)
        except FileNotFoundError:
            raise PromptException(f"Prompts file not found: {self.prompts_path}")
        except yaml.YAMLError as e:
            raise PromptException(f"Invalid YAML in prompts file: {e}")
    
    def get(self, node: str, key: str = "template") -> str:
        """Get prompt template for a specific node.
        
        Args:
            node: Node name (e.g., 'rewrite', 'generate')
            key: Prompt key ('template' or 'system')
            
        Returns:
            Prompt text
            
        Raises:
            PromptException: If node or key not found
        """
        try:
            return self.prompts[node][key]
        except KeyError:
            raise PromptException(f"Prompt not found: node='{node}', key='{key}'")
    
    def get_system(self, node: str) -> str:
        """Get system prompt for a specific node.
        
        Args:
            node: Node name
            
        Returns:
            System prompt text or empty string
        """
        try:
            return self.prompts[node].get("system", "")
        except KeyError:
            return ""
