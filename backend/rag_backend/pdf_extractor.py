"""
PDF Extraction Module using GROBID
Extracts title, abstract, and metadata from research paper PDFs
"""

import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Any
from lxml import etree
import logging

logger = logging.getLogger(__name__)

# GROBID configuration
GROBID_HOST = os.environ.get("GROBID_HOST", "localhost")
GROBID_PORT = os.environ.get("GROBID_PORT", "8070")
GROBID_URL = f"http://{GROBID_HOST}:{GROBID_PORT}"
GROBID_IMAGE = "lfoppiano/grobid:0.8.1"

# XML namespaces for TEI parsing
TEI_NS = {'tei': 'http://www.tei-c.org/ns/1.0'}


class GrobidManager:
    """Manages GROBID Docker container lifecycle"""
    
    @staticmethod
    def is_docker_installed() -> bool:
        """Check if Docker is installed"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    @staticmethod
    def is_grobid_running() -> bool:
        """Check if GROBID is running and responding"""
        try:
            response = requests.get(
                f"{GROBID_URL}/api/isalive",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    @staticmethod
    def start_grobid_container() -> bool:
        """Start GROBID Docker container"""
        if not GrobidManager.is_docker_installed():
            logger.error("Docker is not installed")
            return False
        
        # Check if already running
        if GrobidManager.is_grobid_running():
            logger.info("GROBID is already running")
            return True
        
        try:
            # Check if container exists but is stopped
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=grobid", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if "grobid" in result.stdout:
                # Container exists, start it
                logger.info("Starting existing GROBID container...")
                subprocess.run(
                    ["docker", "start", "grobid"],
                    capture_output=True,
                    timeout=60
                )
            else:
                # Container doesn't exist, create and run it with cgroups v2 compatibility
                logger.info("Creating and starting GROBID container...")
                subprocess.run(
                    [
                        "docker", "run", "-d",
                        "--name", "grobid",
                        "-p", f"{GROBID_PORT}:8070",
                        "-e", "JAVA_OPTS=-XX:-UseContainerSupport",
                        GROBID_IMAGE
                    ],
                    capture_output=True,
                    timeout=120
                )
            
            # Wait for GROBID to be ready
            logger.info("Waiting for GROBID to be ready...")
            for i in range(60):  # Wait up to 60 seconds
                if GrobidManager.is_grobid_running():
                    logger.info("GROBID is ready!")
                    return True
                time.sleep(1)
            
            logger.error("GROBID failed to start within timeout")
            return False
            
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start GROBID container: {e}")
            return False
    
    @staticmethod
    def stop_grobid_container() -> bool:
        """Stop GROBID Docker container"""
        try:
            subprocess.run(
                ["docker", "stop", "grobid"],
                capture_output=True,
                timeout=30
            )
            logger.info("GROBID container stopped")
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to stop GROBID container: {e}")
            return False


class PDFExtractor:
    """Extracts title, abstract, and metadata from PDFs using GROBID"""
    
    def __init__(self, auto_start_grobid: bool = True):
        """
        Initialize PDF extractor
        
        Args:
            auto_start_grobid: Automatically start GROBID if not running
        """
        self.grobid_url = GROBID_URL
        self.auto_start_grobid = auto_start_grobid
        
    def ensure_grobid_running(self) -> bool:
        """Ensure GROBID is running"""
        if GrobidManager.is_grobid_running():
            return True
        
        if self.auto_start_grobid:
            return GrobidManager.start_grobid_container()
        
        return False
    
    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Process PDF file through GROBID and get TEI XML
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            TEI XML string or None if failed
        """
        if not self.ensure_grobid_running():
            raise RuntimeError("GROBID is not available. Please start GROBID first.")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as pdf_file:
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files={'input': pdf_file},
                    data={
                        'consolidateHeader': '1',
                        'consolidateCitations': '0',
                        'includeRawAffiliations': '1'
                    },
                    timeout=120
                )
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"GROBID returned status {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"GROBID request failed: {e}")
            return None
    
    def process_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Optional[str]:
        """
        Process PDF bytes through GROBID and get TEI XML
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            
        Returns:
            TEI XML string or None if failed
        """
        if not self.ensure_grobid_running():
            raise RuntimeError("GROBID is not available. Please start GROBID first.")
        
        try:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={'input': (filename, pdf_bytes, 'application/pdf')},
                data={
                    'consolidateHeader': '1',
                    'consolidateCitations': '0',
                    'includeRawAffiliations': '1'
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"GROBID returned status {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"GROBID request failed: {e}")
            return None
    
    def extract_content(self, xml_content: str) -> Dict[str, Any]:
        """
        Extract title, abstract, and metadata from TEI XML
        
        Args:
            xml_content: TEI XML string from GROBID
            
        Returns:
            Dictionary with extracted content
        """
        result = {
            'title': None,
            'abstract': None,
            'authors': [],
            'year': None,
            'journal': None,
            'doi': None,
            'keywords': []
        }
        
        try:
            root = etree.fromstring(xml_content.encode('utf-8'))
            
            # Extract title
            title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', TEI_NS)
            if title_elem is None:
                title_elem = root.find('.//tei:titleStmt/tei:title', TEI_NS)
            if title_elem is not None:
                result['title'] = self._get_text_content(title_elem)
            
            # Extract abstract
            abstract_elem = root.find('.//tei:profileDesc/tei:abstract', TEI_NS)
            if abstract_elem is not None:
                # Get all text from abstract, including nested elements
                abstract_text = self._get_all_text(abstract_elem)
                result['abstract'] = abstract_text.strip()
            
            # Extract authors
            for author_elem in root.findall('.//tei:fileDesc//tei:author', TEI_NS):
                author_info = self._extract_author(author_elem)
                if author_info:
                    result['authors'].append(author_info)
            
            # Extract publication date/year
            date_elem = root.find('.//tei:publicationStmt/tei:date[@type="published"]', TEI_NS)
            if date_elem is None:
                date_elem = root.find('.//tei:publicationStmt/tei:date', TEI_NS)
            if date_elem is not None:
                when = date_elem.get('when', '')
                if when:
                    result['year'] = int(when[:4]) if len(when) >= 4 else None
            
            # Extract journal/venue
            journal_elem = root.find('.//tei:monogr/tei:title[@level="j"]', TEI_NS)
            if journal_elem is not None:
                result['journal'] = self._get_text_content(journal_elem)
            
            # Extract DOI
            doi_elem = root.find('.//tei:idno[@type="DOI"]', TEI_NS)
            if doi_elem is not None:
                result['doi'] = self._get_text_content(doi_elem)
            
            # Extract keywords
            for keyword_elem in root.findall('.//tei:keywords//tei:term', TEI_NS):
                keyword = self._get_text_content(keyword_elem)
                if keyword:
                    result['keywords'].append(keyword)
            
        except etree.XMLSyntaxError as e:
            logger.error(f"XML parsing error: {e}")
        
        return result
    
    def _get_text_content(self, element) -> str:
        """Get text content of an element"""
        if element is None:
            return ""
        return ''.join(element.itertext()).strip()
    
    def _get_all_text(self, element) -> str:
        """Get all text content including nested elements"""
        texts = []
        for text in element.itertext():
            texts.append(text.strip())
        return ' '.join(filter(None, texts))
    
    def _extract_author(self, author_elem) -> Optional[str]:
        """Extract author name from author element"""
        persname = author_elem.find('.//tei:persName', TEI_NS)
        if persname is None:
            return None
        
        forename = persname.find('.//tei:forename', TEI_NS)
        surname = persname.find('.//tei:surname', TEI_NS)
        
        parts = []
        if forename is not None:
            parts.append(self._get_text_content(forename))
        if surname is not None:
            parts.append(self._get_text_content(surname))
        
        return ' '.join(parts) if parts else None
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete extraction: process PDF and extract content
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        xml_content = self.process_pdf(pdf_path)
        if xml_content:
            return self.extract_content(xml_content)
        return {
            'title': None,
            'abstract': None,
            'authors': [],
            'year': None,
            'journal': None,
            'doi': None,
            'keywords': [],
            'error': 'Failed to process PDF'
        }
    
    def extract_from_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        Complete extraction: process PDF bytes and extract content
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with extracted content
        """
        xml_content = self.process_pdf_bytes(pdf_bytes, filename)
        if xml_content:
            return self.extract_content(xml_content)
        return {
            'title': None,
            'abstract': None,
            'authors': [],
            'year': None,
            'journal': None,
            'doi': None,
            'keywords': [],
            'error': 'Failed to process PDF'
        }


# Convenience function
def extract_title_abstract(pdf_path: str) -> Dict[str, str]:
    """
    Quick extraction of title and abstract from PDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with 'title' and 'abstract'
    """
    extractor = PDFExtractor()
    result = extractor.extract_from_pdf(pdf_path)
    return {
        'title': result.get('title', ''),
        'abstract': result.get('abstract', '')
    }


if __name__ == "__main__":
    # Test extraction
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extractor = PDFExtractor()
        result = extractor.extract_from_pdf(pdf_path)
        
        print("\n=== Extraction Result ===")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Abstract: {result.get('abstract', 'N/A')[:500]}...")
        print(f"Authors: {', '.join(result.get('authors', []))}")
        print(f"Year: {result.get('year', 'N/A')}")
        print(f"Journal: {result.get('journal', 'N/A')}")
        print(f"DOI: {result.get('doi', 'N/A')}")
    else:
        print("Usage: python pdf_extractor.py <pdf_path>")
