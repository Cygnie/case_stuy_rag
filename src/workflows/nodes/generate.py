"""Generate node for answer generation."""
import logging
from src.core.state import GraphState
from src.core.interfaces import BaseLLMService
from src.core.prompts import PromptManager

logger = logging.getLogger(__name__)


class GenerateNode:
    """Node responsible for generating final answer from context."""
    
    def __init__(self, llm: BaseLLMService, prompt_manager: PromptManager):
        """Initialize generate node.
        
        Args:
            llm: LLM service for answer generation
            prompt_manager: Prompt manager for templates
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
    
    def execute(self, state: GraphState) -> GraphState:
        """Generate answer based on retrieved documents.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with answer
        """
        logger.debug(f"Generating answer with {len(state['documents'])} context documents")
        
        system_prompt = self.prompt_manager.get_system("generate")
        template = self.prompt_manager.get("generate", "template")
        
        context = "\n\n".join(state["documents"])
        prompt = template.format(
            context=context,
            question=state["question"]
        )
        
        answer = self.llm.generate(prompt, system=system_prompt)
        state["answer"] = answer.strip()
        
        logger.info(f"Generated answer: {len(state['answer'])} chars")
        return state
