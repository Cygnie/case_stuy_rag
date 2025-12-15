"""Dependency providers for dependency injection."""
from fastapi import Request
from src.core.config import settings, Settings
from src.core.prompts import PromptManager
from src.services.llm import GeminiLLMService
from src.services.vector_store import QdrantVectorStore
from src.services.rag_service import RAGService


def get_settings() -> Settings:
    """Get application settings (Singleton)."""
    return settings


def get_llm_service(request: Request) -> GeminiLLMService:
    """Get LLM service from app.state."""
    return request.app.state.llm_service


def get_vector_service(request: Request) -> QdrantVectorStore:
    """Get vector store service from app.state."""
    return request.app.state.vector_store


def get_rag_service(request: Request) -> RAGService:
    """Get RAG service from app.state."""
    return request.app.state.rag_service


def get_prompt_manager(request: Request) -> PromptManager:
    """Get PromptManager from app.state."""
    return request.app.state.prompt_manager
