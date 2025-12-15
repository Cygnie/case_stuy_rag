"""RAG question answering endpoint - Clean controller layer (ASYNC)."""
import logging
from fastapi import APIRouter, Depends

from src.models.schemas import QueryRequest, QueryResponse
from src.api.dependencies import get_rag_service
from src.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Ask a question using RAG workflow (ASYNC)."""
    # Async call to service layer - doesn't block other requests
    result = await rag_service.ask(request.question)
    
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        rewritten_question=result.rewritten_question,
        years_extracted=result.years_extracted
    )
