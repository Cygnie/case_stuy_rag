import os
import uuid
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Main Libraries
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastembed import SparseTextEmbedding
from langchain_core.documents import Document as LCDocument

# Local Processors (Keep these as they are logic encapsulations)
from src.utils import setup_logger
from src.data_ingestion.scrappers.scraper import Scraper
from src.data_ingestion.processor.pdf_extractor import PDFExtractor
from src.data_ingestion.processor.text_cleaner import TextCleaner
from src.data_ingestion.processor.chunker import ChunkerFactory

logger = setup_logger(__name__)
load_dotenv()

class IngestionService:
    """
    Standalone Ingestion Service.
    Directly uses QdrantClient, GoogleGenerativeAIEmbeddings, and FastEmbed.
    """
    
    def __init__(
        self, 
        raw_data_dir: str = "data/raw", 
        processed_data_dir: str = "data/processed",
        chunking_strategy: str = "recursive"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # 1. Initialize Processors
        self.scraper = Scraper(base_dir=str(self.raw_data_dir))
        self.extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = ChunkerFactory.create(strategy=chunking_strategy)

        # 2. Initialize Infrastructure (Directly)
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "ntt_hybrid_experiment")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # 3. Initialize Embeddings
        google_api_key = os.getenv("EMBEDDING_API_KEY")
        if not google_api_key:
            raise ValueError("EMBEDDING_API_KEY not found in environment variables.")
            
        logger.info("Initializing Gemini Embeddings...")
        self.dense_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key,
            task_type="semantic_similarity"
        )
        
        logger.info("Initializing Sparse Embeddings (BM25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # 4. Ensure Collection Exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating collection '{self.collection_name}'...")
            
            # Get vector size dynamically
            dummy_vec = self.dense_model.embed_query("test")
            vector_size = len(dummy_vec)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            logger.info("Collection created.")
            
            # Create Payload Index for Year
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            logger.info("Payload index for 'year' created.")

    def run(self):
        """Executes the full ingestion pipeline."""
        logger.info("Starting ingestion pipeline...")
        
        # 1. Scrape Data
        logger.info("Step 1: Scraping data...")
        self.scraper.scrape()
        
        # 2. Process & Index Data
        logger.info("Step 2: Processing, Chunking, and Indexing data...")
        self._process_and_index_files()
        
        logger.info("Ingestion pipeline completed successfully.")

    def _process_and_index_files(self):
        """Walks through raw directory, processes files, chunks, and indexes them."""
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_path = Path(root) / file
                    try:
                        self._process_single_file(pdf_path)
                    except Exception as e:
                        logger.error(f"Failed to process {file}: {e}")

    def _process_single_file(self, pdf_path: Path):
        """Processes a single PDF file through the pipeline."""
        logger.info(f"Processing {pdf_path.name}...")

        # A. Extract
        raw_text = self.extractor.extract_text(str(pdf_path))
        if not raw_text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return

        # B. Clean
        cleaned_text = self.cleaner.clean_text(raw_text)
        self._save_processed_text(pdf_path, cleaned_text)

        # C. Chunk
        lc_chunks: List[LCDocument] = self.chunker.chunk_text(cleaned_text)
        if not lc_chunks:
            logger.warning(f"No chunks generated for {pdf_path.name}")
            return
            
        # D. Index
        self._index_chunks(lc_chunks, pdf_path)

    def _index_chunks(self, chunks: List[LCDocument], source_path: Path):
        """Indexes chunks into Qdrant using Hybrid Search (Dense + Sparse)."""
        logger.info(f"Indexing {len(chunks)} chunks for {source_path.name}...")
        
        # Extract Year
        year = self._extract_year(source_path)
        
        texts = [doc.page_content for doc in chunks]
        metadatas = []
        
        for doc in chunks:
            meta = doc.metadata.copy()
            meta["source"] = source_path.name
            meta["file_path"] = str(source_path)
            if year:
                meta["year"] = year
            metadatas.append(meta)

        # Generate Embeddings
        dense_vectors = self.dense_model.embed_documents(texts)
        sparse_vectors = list(self.sparse_model.embed(texts))
        
        points = []
        for i, (text, meta, dense, sparse) in enumerate(zip(texts, metadatas, dense_vectors, sparse_vectors)):
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist()
                    )
                },
                payload={
                    "content": text,
                    **meta
                }
            ))
            
        # Upsert to Qdrant (Sync)
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"Successfully indexed {len(points)} points.")

    def _extract_year(self, source_path: Path) -> int | None:
        """Extracts year from folder structure or filename."""
        try:
            parent_folder = source_path.parent.name
            if "ntt_" in parent_folder:
                year_str = parent_folder.replace("ntt_", "")
                return int(year_str) if year_str.isdigit() else None
            
            import re
            match = re.search(r'20\d{2}', source_path.name)
            return int(match.group(0)) if match else None
        except Exception:
            return None

    def _save_processed_text(self, original_path: Path, text: str):
        """Saves intermediate processed text to disk."""
        rel_path = original_path.relative_to(self.raw_data_dir)
        output_dir = self.processed_data_dir / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{original_path.stem}.md"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
