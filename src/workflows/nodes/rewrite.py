"""Rewrite node for query optimization and year extraction using structured output."""
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.state import GraphState
from src.core.interfaces import BaseLLMService
from src.core.prompts import PromptManager

logger = logging.getLogger(__name__)


class RewriteOutput(BaseModel):
    """Structured output schema for rewrite node."""
    
    years: list[int] = Field(
        default_factory=list,
        description="List of years extracted from the question. Empty list if no years mentioned."
    )
    query: str = Field(
        description="Rewritten query in English, optimized for search"
    )


class RewriteNode:
    """Node responsible for rewriting queries and extracting year filters."""
    
    def __init__(self, llm: BaseLLMService, prompt_manager: PromptManager):
        """Initialize rewrite node."""
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.structured_llm = llm.get_structured_llm(RewriteOutput)
    
    def execute(self, state: GraphState) -> GraphState:
        """Rewrite the user's question and extract years using structured output."""
        system_prompt = self.prompt_manager.get_system("rewrite")
        template = self.prompt_manager.get("rewrite", "template")
        
        prompt = template.format(question=state["question"])
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
        
        try:
            result: RewriteOutput = self.structured_llm.invoke(messages)
            
            logger.debug(f"Structured output: years={result.years}, query='{result.query}'")
            
            state["rewritten_question"] = result.query
            state["years"] = result.years if result.years else None
            
            logger.info(f"Extracted years: {result.years}, Rewritten query: '{result.query[:50]}...'")
            
        except Exception as e:
            logger.warning(f"Structured output failed, falling back: {e}")
            state["rewritten_question"] = state["question"]
            state["years"] = None
        
        return state
