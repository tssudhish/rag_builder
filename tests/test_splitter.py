import pytest
from rag_builder.ingestion.loaders import Document
from rag_builder.ingestion.splitter import TextSplitter, Chunk, TextSegment

def test_splitter_basic_splitting():
    text = "Alpha bravo charlie delta. Echo foxtrot golf hotel. India juliett kilo lima."
    chunk_size = 35
    chunk_overlap = 10
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    
    # 1. Verify content matches exact slice
    for chunk in chunks:
        assert isinstance(chunk, TextSegment)
        assert chunk.content == text[chunk.start:chunk.end]
        assert len(chunk.content) > 0

    # 2. Verify no characters are lost (contiguous coverage and length reconstruction)
    reconstructed_len = len(chunks[0].content)
    for i in range(1, len(chunks)):
        # Next chunk must start at or before prior chunk's end
        assert chunks[i].start <= chunks[i-1].end
        # Overlap content must match exactly
        overlap_len = chunks[i-1].end - chunks[i].start
        assert overlap_len > 0
        assert chunks[i-1].content[-overlap_len:] == chunks[i].content[:overlap_len]
        reconstructed_len += (len(chunks[i].content) - overlap_len)
        
    assert reconstructed_len == len(text)

def test_splitter_boundary_preservation():
    text = "First sentence here. Second sentence starts here. Third sentence arrives now."
    # Sentence 1 ends at index 20. With chunk_size=32, the raw character limit lands at index 32
    # in the middle of the word 'sentence'. With chunk_overlap=15, the search window is [17, 32].
    # It must prioritize and contract to the period at index 20 rather than cutting at index 32.
    splitter = TextSplitter(chunk_size=32, chunk_overlap=15)
    chunks = splitter.split_text(text)
    
    assert len(chunks) >= 2
    # Verify first chunk contracted to sentence boundary rather than cutting at index 32
    assert chunks[0].end == 20
    assert chunks[0].content == "First sentence here."
    assert len(chunks[0].content) < 32
    assert chunks[0].content.endswith(".")

    # Verify second chunk captures subsequent text without dropping letters
    overlap_len = chunks[0].end - chunks[1].start
    assert chunks[1].content[:overlap_len] == chunks[0].content[-overlap_len:]

def test_splitter_overlap_precision():
    text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunk_size = 15
    chunk_overlap = 5
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        overlap_size = c1.end - c2.start
        assert overlap_size == chunk_overlap
        assert c1.content[-overlap_size:] == c2.content[:overlap_size]

def test_splitter_text_smaller_than_chunk_size():
    """Verify that text smaller than chunk_size produces exactly one chunk with full content."""
    text = "Short text under chunk size."
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)

def test_splitter_empty_text():
    splitter = TextSplitter()
    assert splitter.split_text("") == []

def test_splitter_overlap_error():
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        TextSplitter(chunk_size=100, chunk_overlap=100)

def test_split_document():
    doc = Document(content="Hello world. This is a test document.", metadata={"source": "test"}, source="test.txt")
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
    chunks = splitter.split_document(doc)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].source == "test.txt"
    assert chunks[0].metadata["source"] == "test"
    assert chunks[0].char_offset == 0
    assert len(chunks[0].content) > 0

def test_splitter_monotonic_advance():
    text = "Short text."
    splitter = TextSplitter(chunk_size=5, chunk_overlap=2)
    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    for i in range(1, len(chunks)):
        assert chunks[i].start > chunks[i-1].start
