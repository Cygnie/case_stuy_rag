"""
Ingestion Script
Refactored to use the Clean Architecture IngestionService.
"""
import sys
import os

# Add project root to python path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_ingestion.pipeline import IngestionService
from src.utils import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Initializing Ingestion Pipeline...")
    
    try:
        # No complex dependency injection needed anymore
        service = IngestionService()
        
        logger.info("Running Ingestion Service...")
        service.run()
        
        logger.info("Ingestion complete.")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
