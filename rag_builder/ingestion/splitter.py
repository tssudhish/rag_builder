import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, NamedTuple
from rag_builder.ingestion.loaders import Document

class TextSegment(NamedTuple):
    """Represents a sliced text segment with exact offsets."""
    content: str
    start: int
    end: int

@dataclass
class Chunk:
    """Represents a chunk of text extracted from a Document."""
    content: str
    chunk_id: int
    source: Optional[str] = None
    char_offset: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class TextSplitter:
    """Handles splitting of documents into smaller, context-aware chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[TextSegment]:
        """Splits text into chunks based on character count while attempting 
        to preserve sentence boundaries. Returns List[TextSegment].
        """
        if not text:
            return []

        chunks: List[TextSegment] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            curr_start = start
            end = min(start + self.chunk_size, text_len)
            
            if end < text_len:
                # Search for the nearest sentence or line boundary within overlap
                search_start = max(start + 1, end - self.chunk_overlap)
                for i in range(end, search_start, -1):
                    if text[i-1] in {'.', '!', '?'}:
                        if i == text_len or text[i].isspace():
                            end = i
                            break
                    elif text[i-1] == '\n':
                        end = i
                        break
            
            chunk_content = text[start:end]
            if chunk_content.strip():
                chunks.append(TextSegment(content=chunk_content, start=start, end=end))
            
            if end >= text_len:
                break
            
            # Guaranteed strictly monotonic advance per reviewer recommendation
            new_start = end - self.chunk_overlap
            start = max(new_start, curr_start + 1)

        return chunks

    def split_document(self, doc: Document) -> List[Chunk]:
        """Splits a Document into a list of Chunks.
        
        Args:
            doc: The Document to split.
            
        Returns:
            A list of Chunk objects with associated metadata.
        """
        text_chunks = self.split_text(doc.content)
        chunks = []
        
        for idx, (content, start, end) in enumerate(text_chunks):
            chunks.append(Chunk(
                content=content,
                chunk_id=idx,
                source=doc.source,
                char_offset=start,
                metadata=doc.metadata.copy()
            ))
            
        return chunks
