from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

import PyPDF2
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a loaded document with its content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

class BaseLoader(ABC):
    """Abstract base class for all document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Document:
        """Loads a document from the given file path.
        
        Args:
            file_path: Path to the file to be loaded.
            
        Returns:
            A Document object containing the text and metadata.
        """
        pass

class PlainTextLoader(BaseLoader):
    """Base loader for text-based files with encoding fallbacks."""
    
    def __init__(self, file_type: str = "text"):
        self.file_type = file_type

    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                content = path.read_text(encoding='latin-1', errors='replace')
            return Document(
                content=content,
                metadata={"file_type": self.file_type},
                source=str(path)
            )
        except (IOError, PermissionError) as e:
            logger.error(f"Error loading {self.file_type} file {file_path}: {e}")
            raise

class TextLoader(PlainTextLoader):
    """Loader for plain text (.txt) files."""
    def __init__(self):
        super().__init__(file_type="text")

class MarkdownLoader(PlainTextLoader):
    """Loader for Markdown (.md) files."""
    def __init__(self):
        super().__init__(file_type="markdown")

class PDFLoader(BaseLoader):
    """Loader for PDF files."""
    
    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            clean_text = text.strip()
            if not clean_text:
                logger.warning(f"PDFLoader: No text extracted from {file_path}. Document might be scanned or image-based.")

            return Document(
                content=clean_text,
                metadata={"file_type": "pdf", "pages": len(reader.pages)},
                source=str(path)
            )
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            raise

class HTMLLoader(BaseLoader):
    """Loader for HTML files."""
    
    def load(self, file_path: str) -> Document:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            try:
                html_content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                html_content = path.read_text(encoding='latin-1', errors='replace')
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
                
            raw_text = soup.get_text(separator=' ')
            # Clean up redundant whitespace while preserving paragraph lines
            cleaned_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            text = '\n'.join(cleaned_lines)
            
            return Document(
                content=text,
                metadata={"file_type": "html"},
                source=str(path)
            )
        except (IOError, PermissionError) as e:
            logger.error(f"Error loading HTML file {file_path}: {e}")
            raise

# Loader instances for the factory
LOADERS_MAPPING = {
    '.txt': TextLoader(),
    '.md': MarkdownLoader(),
    '.pdf': PDFLoader(),
    '.html': HTMLLoader(),
    '.htm': HTMLLoader(),
}

def load_document(file_path: str) -> Document:
    """Factory function to load a document based on its extension."""
    ext = Path(file_path).suffix.lower()
    loader = LOADERS_MAPPING.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file extension: {ext}")
    return loader.load(file_path)
