"""Hybrid vector store using Qdrant with dense and sparse vectors."""
import logging
from typing import Optional
from qdrant_client import QdrantClient, models

from src.core.interfaces import BaseVectorStore
from src.core.exceptions import VectorStoreException
from src.services.embeddings import GeminiEmbeddingService, FastEmbedSparseService

logger = logging.getLogger(__name__)



class QdrantVectorStore(BaseVectorStore):
    """Hybrid search vector store using Qdrant client."""
    
    def __init__(
        self,
        client: QdrantClient,
        dense_embedder: GeminiEmbeddingService,
        sparse_embedder: FastEmbedSparseService,
        collection_name: str = "documents"
    ):
        """Initialize Qdrant vector store."""
        try:
            self.client = client
            self.dense_embedder = dense_embedder
            self.sparse_embedder = sparse_embedder
            self.collection_name = collection_name
            self._ensure_collection()
            logger.info(f"Vector store initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreException(f"Failed to initialize vector store: {e}") from e
            
    def _convert_sparse_embedding(self, sparse_embedding) -> models.SparseVector:
        """Convert fastembed SparseEmbedding to Qdrant SparseVector."""
        return models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )
    
    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name in collection_names:
            logger.debug(f"Collection '{self.collection_name}' already exists")
            return
        
        logger.info(f"Creating new collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            }
        )
    
    def add_documents(self, docs: list[dict]) -> None:
        """Add documents to the vector store."""
        logger.info(f"Adding {len(docs)} documents to vector store")
        try:
            points = []
            for idx, doc in enumerate(docs):
                dense_vector = self.dense_embedder.embed(doc["content"])
                sparse_embedding = self.sparse_embedder.embed(doc["content"])
                sparse_vector = self._convert_sparse_embedding(sparse_embedding)
                
                points.append(models.PointStruct(
                    id=idx,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector
                    },
                    payload={
                        "content": doc["content"],
                        "source": doc.get("source", ""),
                        "year": doc.get("year"),
                        "chunk_index": doc.get("chunk_index", 0)
                    }
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully added {len(docs)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {type(e).__name__}: {e}")
            raise VectorStoreException(f"Failed to add documents: {e}") from e
    
    def _build_year_filter(self, years: Optional[list[int]]) -> Optional[models.Filter]:
        """Build Qdrant filter for years."""
        if not years:
            return None
        
        logger.debug(f"Building year filter: {years}")
        return models.Filter(
            should=[
                models.FieldCondition(
                    key="year",
                    match=models.MatchValue(value=year)
                )
                for year in years
            ]
        )
    
    def search(self, query: str, k: int = 4) -> list[str]:
        """Simple search using only dense embeddings (cosine similarity).
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of document contents
        """
        logger.debug(f"Simple search: '{query[:50]}...', k: {k}")
        
        try:
            dense_vec = self.dense_embedder.embed(query)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec,
                using="dense",
                limit=k
            )
            
            documents = [hit.payload["content"] for hit in results.points]
            logger.info(f"Simple search retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Simple search failed: {type(e).__name__}: {e}")
            raise VectorStoreException(f"Search failed: {e}") from e
    
    def advanced_search(self, query: str, years: Optional[list[int]] = None, k: int = 4) -> list[str]:
        """Advanced hybrid search using dense + sparse embeddings with RRF fusion.
        
        Args:
            query: Search query
            years: Optional list of years to filter
            k: Number of results
            
        Returns:
            List of document contents
        """
        logger.debug(f"Advanced search: '{query[:50]}...', years: {years}, k: {k}")
        
        try:
            dense_vec = self.dense_embedder.embed(query)
            sparse_embedding = self.sparse_embedder.embed(query)
            sparse_vec = self._convert_sparse_embedding(sparse_embedding)
            query_filter = self._build_year_filter(years)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using="dense",
                        filter=query_filter,
                        limit=k * 2
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        filter=query_filter,
                        limit=k * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k
            )
            
            documents = [hit.payload["content"] for hit in results.points]
            logger.info(f"Advanced search retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Advanced search failed: {type(e).__name__}: {e}")
            raise VectorStoreException(f"Advanced search failed: {e}") from e
