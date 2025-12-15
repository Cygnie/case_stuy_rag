import os
from docling.document_converter import DocumentConverter
from src.utils import setup_logger

logger = setup_logger(__name__)

class PDFExtractor:
    def __init__(self):
        self.converter = DocumentConverter()

    def extract_text(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file using Docling and returns it as Markdown.
        """
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return ""

        try:
            logger.info(f"Converting '{pdf_path}'...")
            result = self.converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully converted '{pdf_path}'.")
            return markdown_content
        except Exception as e:
            logger.error(f"Error converting '{pdf_path}': {e}")
            return ""
