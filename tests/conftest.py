"""Pytest configuration and fixtures."""
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock
from typing import Optional

# Set test environment variables BEFORE importing anything
os.environ["LLM_API_KEY"] = "test-api-key"
os.environ["EMBEDDING_API_KEY"] = "test-api-key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["QDRANT_COLLECTION_NAME"] = "test_collection"

from src.core.config import Settings
from src.core.interfaces import BaseLLMService, BaseVectorStore, BaseEmbeddingService
from src.services.rag_service import RAGResponse
from src.api.v1.router import api_router


# ============================================================
# Mock Services - Implement interfaces for proper OOP
# ============================================================

class MockLLMService(BaseLLMService):
    """Mock LLM service implementing BaseLLMService interface."""
    
    def generate(self, prompt: str, system: str = "") -> str:
        """Return mock response."""
        return "Mock answer based on provided context."
    
    def get_structured_llm(self, schema):
        """Return mock structured LLM."""
        from unittest.mock import Mock
        mock_llm = Mock()
        # Return a mock that produces a valid RewriteOutput-like object
        mock_result = Mock()
        mock_result.years = [2023]
        mock_result.query = "Mock rewritten query"
        mock_llm.invoke.return_value = mock_result
        return mock_llm


class MockEmbeddingService(BaseEmbeddingService):
    """Mock embedding service implementing BaseEmbeddingService interface."""
    
    def embed(self, text: str) -> list[float]:
        """Return mock embedding vector."""
        return [0.1] * 768  # Mock 768-dim vector


class MockVectorStore(BaseVectorStore):
    """Mock vector store implementing BaseVectorStore interface."""
    
    def __init__(self):
        self.client = Mock()
        self.client.get_collections.return_value = []
        self.documents = []
    
    def add_documents(self, docs: list[dict]) -> None:
        """Store mock documents."""
        self.documents.extend(docs)
    
    def search(self, query: str, k: int = 4) -> list[str]:
        """Simple search - cosine similarity only."""
        return ["Mock document 1: NTT DATA sustainability.", "Mock document 2: Carbon neutrality by 2030."]
    
    def advanced_search(self, query: str, years: Optional[list[int]] = None, k: int = 4) -> list[str]:
        """Advanced hybrid search with RRF fusion."""
        return ["Mock document 1: NTT DATA sustainability.", "Mock document 2: Carbon neutrality by 2030."]


class MockPromptManager:
    """Mock prompt manager for testing."""
    
    def get(self, node: str, key: str = "template") -> str:
        """Return mock template."""
        if node == "rewrite":
            return "Rewrite this: {question}"
        return "Context: {context}\nQuestion: {question}"
    
    def get_system(self, node: str) -> str:
        """Return empty system prompt."""
        return ""


class MockRAGService:
    """Mock RAG service for integration testing."""
    
    async def ask(self, question: str) -> RAGResponse:
        """Return mock RAG response (async)."""
        return RAGResponse(
            answer="This is a mock answer for testing purposes.",
            sources=["mock_source_1.pdf", "mock_source_2.pdf"],
            rewritten_question="What is the mock question in English?",
            years_extracted=[2023, 2024]
        )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings()


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService()


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    return MockVectorStore()


@pytest.fixture
def mock_prompt_manager():
    """Create mock prompt manager."""
    return MockPromptManager()


@pytest.fixture
def client(mock_settings):
    """Create test client with mocked services.
    
    Creates a fresh FastAPI app WITHOUT lifespan to avoid real service init.
    Uses container pattern for consistency with production code.
    """
    from src.container.container import ServiceContainer
    
    # Create a fresh app without lifespan
    test_app = FastAPI()
    test_app.include_router(api_router, prefix="/api/v1")
    
    # Add root endpoint
    @test_app.get("/")
    async def root():
        return {"message": "Test API"}
    
    # Create mock container
    container = ServiceContainer()
    container.llm_service = MockLLMService()
    container.vector_store = MockVectorStore()
    container.rag_service = MockRAGService()
    
    # Set container in app state (new pattern)
    test_app.state.container = container
    
    with TestClient(test_app) as test_client:
        yield test_client
