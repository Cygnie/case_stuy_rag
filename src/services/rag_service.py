"""RAG Service - Business logic layer for RAG operations (ASYNC)."""
import logging
from dataclasses import dataclass
from typing import Optional

from src.core import BaseLLMService, BaseVectorStore, GraphState
from src.prompts.prompts import PromptManager
from src.workflows.graph import RAGGraph

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG service."""
    answer: str
    sources: list[str]
    rewritten_question: Optional[str] = None
    years_extracted: Optional[list[int]] = None


class RAGService:
    """Business logic layer for RAG operations (ASYNC).
    
    Uses RAGGraph which compiles workflow once and caches it.
    """
    
    def __init__(
        self,
        llm: BaseLLMService,
        vector_store: BaseVectorStore,
        prompt_manager: PromptManager,
        rag_k: int = 5
    ):
        """Initialize RAG service with dependencies.
        
        Args:
            llm: LLM service for text generation
            vector_store: Vector store for document retrieval
            prompt_manager: Prompt template manager
            rag_k: Number of documents to retrieve
        """
        # Create RAG graph (will compile on first use)
        self.graph = RAGGraph(
            llm=llm,
            vector_store=vector_store,
            prompt_manager=prompt_manager,
            rag_k=rag_k
        )
        logger.info("RAGService initialized with lazy-compiled graph")
    
    async def ask(self, question: str) -> RAGResponse:
        """Process a question and return answer with sources (ASYNC).
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        logger.info(f"Processing question: '{question[:50]}...'")
        
        # Use graph's run() helper (compiles graph on first call, cached thereafter)
        result = await self.graph.run(question)
        
        response = RAGResponse(
            answer=result["answer"],
            sources=result.get("documents", []),
            rewritten_question=result.get("rewritten_question"),
            years_extracted=result.get("years")
        )
        
        logger.info(f"Generated answer with {len(response.sources)} sources")
        return response
