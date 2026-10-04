import pytest
import os
from pathlib import Path
from rag_builder.ingestion.loaders import load_document, TextLoader, MarkdownLoader, PDFLoader, HTMLLoader, Document

@pytest.fixture
def sample_files(tmp_path):
    """Creates temporary sample files for testing all loaders in isolation."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    
    txt_file = samples_dir / "test.txt"
    txt_file.write_text("Hello World from TXT", encoding="utf-8")
    
    md_file = samples_dir / "test.md"
    md_file.write_text("# Hello World from MD", encoding="utf-8")
    
    html_file = samples_dir / "test.html"
    html_file.write_text("<html><body><h1>Hello World from HTML</h1><script>alert('bad')</script></body></html>", encoding="utf-8")
    
    # Use static committed sample PDF asset
    fixture_sample = Path(__file__).parent / "samples" / "sample.pdf"
    pdf_file = samples_dir / "test.pdf"
    if fixture_sample.exists():
        pdf_file.write_bytes(fixture_sample.read_bytes())
            
    return samples_dir

def test_text_loader(sample_files):
    file_path = str(sample_files / "test.txt")
    doc = load_document(file_path)
    assert isinstance(doc, Document)
    assert "Hello World from TXT" in doc.content
    assert doc.metadata["file_type"] == "text"

def test_markdown_loader(sample_files):
    file_path = str(sample_files / "test.md")
    doc = load_document(file_path)
    assert isinstance(doc, Document)
    assert "# Hello World from MD" in doc.content
    assert doc.metadata["file_type"] == "markdown"

def test_html_loader(sample_files):
    file_path = str(sample_files / "test.html")
    doc = load_document(file_path)
    assert isinstance(doc, Document)
    assert "Hello World from HTML" in doc.content
    assert "alert" not in doc.content  # Verifies script tag removal
    assert doc.metadata["file_type"] == "html"

def test_pdf_loader(sample_files):
    pdf_path = sample_files / "test.pdf"
    if not pdf_path.exists():
        pytest.skip("Test PDF could not be generated")
    
    doc = load_document(str(pdf_path))
    assert isinstance(doc, Document)
    assert "sample pdf file" in doc.content.lower()
    assert doc.metadata["file_type"] == "pdf"

def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document("test.exe")

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_file.txt")
