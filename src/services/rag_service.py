"""RAG Service - Business logic layer for RAG operations (ASYNC)."""
import logging
from dataclasses import dataclass
from typing import Optional

from src.core import BaseLLMService, BaseVectorStore, PromptManager, GraphState
from src.workflows.nodes import RewriteNode, RetrieveNode, GenerateNode
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
    
    Orchestrates the RAG workflow: Rewrite -> Retrieve -> Generate.
    Uses async to avoid blocking the event loop.
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
        self.llm = llm
        self.vector_store = vector_store
        self.prompt_manager = prompt_manager
        self.rag_k = rag_k
        self.graph = self._build_graph()
        logger.info("RAGService initialized (async)")
    
    def _build_graph(self):
        """Build and compile the LangGraph workflow.
        
        Returns:
            Compiled LangGraph application
        """
        rewrite_node = RewriteNode(
            llm=self.llm,
            prompt_manager=self.prompt_manager
        )
        retrieve_node = RetrieveNode(
            vector_store=self.vector_store,
            k=self.rag_k
        )
        generate_node = GenerateNode(
            llm=self.llm,
            prompt_manager=self.prompt_manager
        )
        
        rag_graph = RAGGraph(
            rewrite_node=rewrite_node,
            retrieve_node=retrieve_node,
            generate_node=generate_node
        )
        
        logger.debug("RAG graph compiled")
        return rag_graph.compile()
    
    async def ask(self, question: str) -> RAGResponse:
        """Process a question and return answer with sources (ASYNC).
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        logger.info(f"Processing question: '{question[:50]}...'")
        
        initial_state: GraphState = {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        response = RAGResponse(
            answer=result["answer"],
            sources=result.get("documents", []),
            rewritten_question=result.get("rewritten_question"),
            years_extracted=result.get("years")
        )
        
        logger.info(f"Generated answer with {len(response.sources)} sources")
        return response
