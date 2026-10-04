"""Tests for rag_builder.ingestion.loaders."""
import unittest
import tempfile
from pathlib import Path
from rag_builder.ingestion.loaders import (
    Document,
    TextLoader,
    MarkdownLoader,
    HTMLLoader,
    load_document
)

class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_text_loader(self):
        f = self.dir_path / "sample.txt"
        f.write_text("Hello, world! This is a test file.", encoding="utf-8")
        doc = load_document(str(f))
        self.assertEqual(doc.content, "Hello, world! This is a test file.")
        self.assertEqual(doc.metadata["file_type"], "text")
        self.assertEqual(doc.source, str(f))

    def test_markdown_loader(self):
        f = self.dir_path / "sample.md"
        f.write_text("# Title\n\nParagraph text.", encoding="utf-8")
        doc = load_document(str(f))
        self.assertIn("# Title", doc.content)
        self.assertEqual(doc.metadata["file_type"], "markdown")

    def test_html_loader(self):
        f = self.dir_path / "sample.html"
        html = "<html><body><h1>Header</h1><p>Body text.</p><script>alert(1);</script></body></html>"
        f.write_text(html, encoding="utf-8")
        doc = load_document(str(f))
        self.assertIn("Header", doc.content)
        self.assertIn("Body text.", doc.content)
        self.assertNotIn("alert", doc.content)
        self.assertEqual(doc.metadata["file_type"], "html")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_document(str(self.dir_path / "nonexistent.txt"))

    def test_unsupported_extension_raises(self):
        f = self.dir_path / "file.unknown"
        f.write_text("content", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_document(str(f))

if __name__ == "__main__":
    unittest.main()
