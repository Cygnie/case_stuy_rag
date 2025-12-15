"""Unit tests for vector store operations."""
import pytest
import numpy as np

from src.services.vector_store import QdrantVectorStore


class TestVectorStore:
    """Unit tests for vector store operations."""
    
    def test_convert_sparse_embedding(self):
        """Test sparse embedding conversion."""
        # Create mock sparse embedding
        class MockSparseEmbedding:
            indices = np.array([0, 5, 10])
            values = np.array([0.1, 0.5, 0.9])
        
        # Create instance bypassing __init__
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        result = store._convert_sparse_embedding(MockSparseEmbedding())
        
        assert result.indices == [0, 5, 10]
        assert result.values == [0.1, 0.5, 0.9]
    
    def test_empty_sparse_embedding(self):
        """Test handling of empty sparse embedding."""
        class MockEmptySparseEmbedding:
            indices = np.array([])
            values = np.array([])
        
        # Create instance bypassing __init__
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        result = store._convert_sparse_embedding(MockEmptySparseEmbedding())
        
        assert result.indices == []
        assert result.values == []
    
    def test_sparse_embedding_with_many_values(self):
        """Test sparse embedding with many non-zero values."""
        class MockLargeSparseEmbedding:
            indices = np.array([1, 3, 5, 7, 9, 11, 13])
            values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        result = store._convert_sparse_embedding(MockLargeSparseEmbedding())
        
        assert len(result.indices) == 7
        assert len(result.values) == 7
        assert result.indices == [1, 3, 5, 7, 9, 11, 13]
