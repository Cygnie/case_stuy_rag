"""RAG workflow graph using LangGraph."""
from langgraph.graph import StateGraph, END

from src.core import GraphState
from src.workflows.nodes import RewriteNode, RetrieveNode, GenerateNode


class RAGGraph:
    """Orchestrator for RAG workflow: Rewrite -> Retrieve -> Generate."""
    
    def __init__(
        self,
        rewrite_node: RewriteNode,
        retrieve_node: RetrieveNode,
        generate_node: GenerateNode
    ):
        """Initialize RAG graph.
        
        Args:
            rewrite_node: Node for query rewriting
            retrieve_node: Node for context retrieval
            generate_node: Node for answer generation
        """
        self.rewrite_node = rewrite_node
        self.retrieve_node = retrieve_node
        self.generate_node = generate_node
    
    def compile(self):
        """Compile the LangGraph workflow.
        
        Returns:
            Compiled runnable graph
        """
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
        
        return workflow.compile()
