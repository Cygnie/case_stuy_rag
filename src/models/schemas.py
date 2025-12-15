"""Pydantic schemas for API requests and responses."""
from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for RAG query endpoint.
    
    Attributes:
        question: User's question (years will be auto-extracted by rewrite node)
    """
    question: str = Field(..., min_length=1, description="User question")


class QueryResponse(BaseModel):
    """Response schema for RAG query endpoint.
    
    Attributes:
        answer: Generated answer
        sources: Retrieved source documents
        rewritten_question: Optimized search query
        years_extracted: Years that were extracted from the question
    """
    answer: str
    sources: list[str]
    rewritten_question: Optional[str] = None
    years_extracted: Optional[list[int]] = None
