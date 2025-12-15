"""RAG workflow graph using LangGraph."""
import logging
from langgraph.graph import StateGraph, END
import uuid

from src.core import GraphState, BaseLLMService, BaseVectorStore
from src.prompts.prompts import PromptManager
from src.workflows.nodes.rewrite import RewriteNode
from src.workflows.nodes import RetrieveNode, GenerateNode


class RAGGraph:
    """
    RAG workflow builder using LangGraph.
    Compiles graph once and caches it for reuse ("Compile Once, Run Many").
    """
    
    def __init__(
        self,
        llm: BaseLLMService,
        vector_store: BaseVectorStore,
        prompt_manager: PromptManager,
        rag_k: int = 5
    ):
        """Initialize RAG graph with services.
        
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
        self._compiled = None  # Cache for compiled graph
        
        # Create nodes (reusable)
        self.rewrite_node = RewriteNode(llm=llm, prompt_manager=prompt_manager)
        self.retrieve_node = RetrieveNode(vector_store=vector_store, k=rag_k)
        self.generate_node = GenerateNode(llm=llm, prompt_manager=prompt_manager)
    
    def build(self):
        """Build and compile the LangGraph workflow (cached).
        
        Compiles the graph once and caches it for subsequent calls.
        
        Returns:
            Compiled LangGraph workflow
        """
        if self._compiled:
            return self._compiled
        
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("rewrite", self.rewrite_node.execute)
        workflow.add_node("retrieve", self.retrieve_node.execute)
        workflow.add_node("generate", self.generate_node.execute)
        
        # Define edges: rewrite -> retrieve -> generate -> END
        workflow.set_entry_point("rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile and cache
        self._compiled = workflow.compile()
        return self._compiled
    
    async def run(self, question: str):
        """Helper to run the compiled graph asynchronously.
        
        Args:
            question: User's question
            
        Returns:
            Final state dict with answer, sources, etc.
        """
        graph = self.build()  # Get cached compiled graph
        
        initial_state: GraphState = {
            "question": question,
            "rewritten_question": "",
            "documents": [],
            "answer": "",
            "years": None
        }
        
        result = await graph.ainvoke(initial_state)
        return result
