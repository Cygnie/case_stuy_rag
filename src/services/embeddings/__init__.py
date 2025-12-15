"""Embedding services package with factory pattern."""

from src.services.embeddings.gemini import GeminiEmbeddingService
from src.services.embeddings.fastembed import FastEmbedSparseService
from src.services.embeddings.factory import EmbeddingFactory

__all__ = [
    "GeminiEmbeddingService",
    "FastEmbedSparseService",
    "EmbeddingFactory",
]
