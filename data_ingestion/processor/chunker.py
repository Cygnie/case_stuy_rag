"""
Text Chunker module with factory pattern for different chunking strategies.
"""
from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, 
    RecursiveCharacterTextSplitter
)
from langchain_core.documents import Document


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into chunks and return as Documents."""
        pass


class MarkdownChunker(BaseChunker):
    """
    Chunks text based on Markdown headers. 
    If a section is too large, it recursively splits it further.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        # Secondary splitter for large sections
        self.recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text by markdown headers, then recursively if needed."""
        # 1. Split by Markdown Headers
        header_splits = self.markdown_splitter.split_text(text)
        
        # 2. Split large sections recursively
        final_chunks = self.recursive_splitter.split_documents(header_splits)
        
        return final_chunks


class RecursiveChunker(BaseChunker):
    """Chunks text using recursive character splitting with token awareness."""
    
    # Default configuration from notebook experiments
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 150
    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text recursively by character count with token awareness."""
        chunks = self.splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]


class ChunkerFactory:
    """Factory class for creating chunker instances."""
    
    @staticmethod
    def create(strategy: str = "recursive", **kwargs) -> BaseChunker:
        """
        Create a chunker instance based on the strategy.
        
        Args:
            strategy: The chunking strategy ('markdown' or 'recursive').
            **kwargs: Additional arguments passed to the chunker constructor.
            
        Returns:
            A chunker instance.
            
        Raises:
            ValueError: If the strategy is not supported.
        """
        chunkers = {
            "markdown": MarkdownChunker,
            "recursive": RecursiveChunker,
        }
        
        if strategy not in chunkers:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available strategies: {list(chunkers.keys())}"
            )
        
        return chunkers[strategy](**kwargs)
    
    @classmethod
    def register(cls, name: str, chunker_class: type):
        """Register a new chunker strategy."""
        cls._chunkers[name] = chunker_class


# Backward compatibility alias
class Chunker:
    """
    Legacy wrapper for backward compatibility.
    Use ChunkerFactory.create() for new code.
    """
    
    def __init__(self, strategy: str = "recursive", **kwargs):
        self._chunker = ChunkerFactory.create(strategy, **kwargs)
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks.
        
        Note: chunk_size parameter is ignored if strategy is 'markdown'.
        For 'recursive' strategy, use constructor kwargs instead.
        """
        docs = self._chunker.chunk_text(text)
        return [doc.page_content for doc in docs]
