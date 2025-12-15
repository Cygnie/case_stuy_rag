"""Graph state definition for LangGraph workflow."""
from typing import TypedDict, Optional


class GraphState(TypedDict):
    """State structure for RAG workflow graph.
    
    Attributes:
        question: Original user question
        rewritten_question: Optimized query for search
        documents: Retrieved context documents
        answer: Generated final answer
        years: Optional list of years to filter documents (e.g., [2021, 2022, 2023])
    """
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str
    years: Optional[list[int]]
