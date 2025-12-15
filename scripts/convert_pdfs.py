import os
import sys
import argparse

# Add project root to python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion.processor.pdf_extractor import PDFExtractor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using Docling.")
    parser.add_argument("--input", "-i", required=True, help="Input PDF file or directory containing PDFs")
    parser.add_argument("--output", "-o", required=True, help="Output directory for Markdown files")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Initializing PDF Extractor...")
    extractor = PDFExtractor()
    
    files_to_process = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.pdf'):
            files_to_process.append(input_path)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    files_to_process.append(os.path.join(root, file))
    
    if not files_to_process:
        logger.warning("No PDF files found to process.")
        return
        
    logger.info(f"Found {len(files_to_process)} PDF files. Starting conversion...")
    
    success_count = 0
    error_count = 0
    
    for pdf_file in files_to_process:
        try:
            # Determine output filename
            # We will use the relative path structure if input is a directory, 
            # or just the filename if input is a file.
            
            if os.path.isdir(input_path):
                rel_path = os.path.relpath(pdf_file, input_path)
                rel_dir = os.path.dirname(rel_path)
                target_dir = os.path.join(output_dir, rel_dir)
            else:
                target_dir = output_dir

            os.makedirs(target_dir, exist_ok=True)
            
            base_name = os.path.basename(pdf_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_file_path = os.path.join(target_dir, f"{file_name_without_ext}.md")
            
            logger.info(f"Processing: {pdf_file}")
            markdown_content = extractor.extract_text(pdf_file)
            
            if markdown_content:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Saved to: {output_file_path}")
                success_count += 1
            else:
                logger.warning(f"No content extracted from: {pdf_file}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"Failed to convert {pdf_file}: {e}")
            error_count += 1
            
    logger.info(f"Conversion complete. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()
