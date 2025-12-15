import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import logging
from urllib.parse import urljoin
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Scraper:
    BASE_URL = "https://www.nttdata.com/global/en/sustainability/report"
    
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _make_request(self, url: str, stream: bool = False) -> requests.Response:
        """Helper method to make HTTP requests with retry logic."""
        response = requests.get(url, stream=stream, timeout=30)
        response.raise_for_status()
        return response

    def scrape(self, years_back: int = 5):
        """
        Scrapes PDF reports for the last X years.
        """
        logger.info(f"Starting scrape for the last {years_back} years...")
        
        try:
            response = self._make_request(self.BASE_URL)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            current_year = datetime.now().year
            target_years = range(current_year - years_back, current_year + 1)
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            downloaded_count = 0
            
            for link in links:
                href = link['href']
                text = link.get_text().strip()
                
                # Check if it is a PDF
                if not href.lower().endswith('.pdf') and not '.pdf?' in href.lower():
                    continue
                
                # Determine year
                year = self._extract_year(href, text)
                
                if year and year in target_years:
                    full_url = urljoin(self.BASE_URL, href)
                    self._download_pdf(full_url, year)
                    downloaded_count += 1
            
            logger.info(f"Scraping completed. Downloaded {downloaded_count} files.")
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")

    def _extract_year(self, url: str, text: str) -> int | None:
        """
        Extracts year from URL or link text.
        """
        # Try to find year in URL (e.g., /2024/)
        year_match = re.search(r'/(\d{4})/', url)
        if year_match:
            return int(year_match.group(1))
            
        # Try to find year in text (e.g., "Sustainability Report 2024")
        year_match = re.search(r'20\d{2}', text)
        if year_match:
            return int(year_match.group(0))
            
        return None

    def _download_pdf(self, url: str, year: int):
        """
        Downloads a PDF file to the year-specific directory.
        """
        try:
            year_dir = self.base_dir / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            filename = url.split('/')[-1].split('?')[0] # Remove query params
            file_path = year_dir / filename
            
            if file_path.exists():
                logger.info(f"File already exists: {file_path}")
                return

            logger.info(f"Downloading {url} to {file_path}...")
            response = self._make_request(url, stream=True)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {filename}")
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    scraper = Scraper()
    scraper.scrape(years_back=5)