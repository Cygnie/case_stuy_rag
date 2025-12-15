"""Retrieve node for context retrieval from vector store."""
import logging
from src.core.state import GraphState
from src.core.interfaces import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrieveNode:
    """Node responsible for retrieving relevant context from vector store."""
    
    def __init__(self, vector_store: BaseVectorStore, k: int = 5):
        """Initialize retrieve node.
        
        Args:
            vector_store: Vector store for document retrieval
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.k = k
    
    def execute(self, state: GraphState) -> GraphState:
        """Retrieve relevant documents using the rewritten question.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with documents
        """
        query = state.get("rewritten_question", state["question"])
        years = state.get("years")
        
        logger.debug(f"Retrieving documents for query: '{query[:50]}...', years: {years}")
        
        # Use advanced_search for hybrid dense + sparse with RRF fusion
        documents = self.vector_store.advanced_search(query=query, years=years, k=self.k)
        state["documents"] = documents
        
        logger.info(f"Retrieved {len(documents)} documents")
        return state
