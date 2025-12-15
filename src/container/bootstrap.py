"""Service initialization and container building.

This module creates and wires all application services into a container.
It reads configuration, uses factories to create instances, and assembles
them with proper dependency injection.
"""
import logging
from qdrant_client import QdrantClient

from src.core.config import Settings
from src.core.exceptions import VectorStoreException
from src.container.container import ServiceContainer
from src.core.enums import LLMProvider, EmbeddingProvider
from src.prompts.prompts import PromptManager

from src.services.llm import LLMFactory
from src.services.embeddings import EmbeddingFactory
from src.services.vector_stores import QdrantVectorStore
from src.services.rag_service import RAGService

logger = logging.getLogger(__name__)


def build_container(settings: Settings) -> ServiceContainer:
    """Build and initialize the service container.
    
    This function is the composition root where all services are created
    and wired together. It assumes providers are already registered.
    
    Args:
        settings: Application settings with configuration
        
    Returns:
        Initialized ServiceContainer with all services ready
        
    Raises:
        VectorStoreException: If vector store initialization fails

    """
    logger.info("Building service container...")
    
    # Create container
    container = ServiceContainer()
    
    # Create LLM service using factory
    container.llm_service = LLMFactory.create(
        provider=LLMProvider(settings.llm_provider),
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature
    )
    logger.info(f"LLM service created: {settings.llm_provider}")
    
    # Create embedding services using factory (for consistency)
    dense_embedder = EmbeddingFactory.create(
        provider=EmbeddingProvider.GEMINI,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model
    )
    sparse_embedder = EmbeddingFactory.create(
        provider=EmbeddingProvider.FASTEMBED_SPARSE
    )
    logger.info("Embedding services created")
    
    # Create Qdrant client and vector store
    try:
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        container._qdrant_client = qdrant_client
        logger.info(f"Qdrant connected: {settings.qdrant_url}")
    except Exception as e:
        logger.critical(f"Qdrant connection failed at {settings.qdrant_url}: {e}")
        raise
    
    try:
        container.vector_store = QdrantVectorStore(
            client=qdrant_client,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=settings.qdrant_collection_name
        )
        logger.info("Vector store initialized")
    except VectorStoreException as e:
        logger.critical(f"Vector store init failed (collection={settings.qdrant_collection_name}): {e}")
        raise
    
    # 6. Create prompt manager
    container.prompt_manager = PromptManager()
    logger.info("Prompt manager initialized")
    
    # 7. Create RAG service (main business logic)
    container.rag_service = RAGService(
        llm=container.llm_service,
        vector_store=container.vector_store,
        prompt_manager=container.prompt_manager,
        rag_k=settings.rag_k
    )
    logger.info("RAG service initialized")
    
    logger.info(f"Service container built successfully: {container}")
    return container
