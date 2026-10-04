"""Tests for rag_builder.ingestion.splitter."""
import unittest
from rag_builder.ingestion.loaders import Document
from rag_builder.ingestion.splitter import TextSplitter, Chunk, TextSegment

class TestTextSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = TextSplitter(chunk_size=50, chunk_overlap=15)

    def test_empty_string(self):
        self.assertEqual(self.splitter.split_text(""), [])

    def test_short_string_single_chunk(self):
        text = "Short text."
        chunks = self.splitter.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, text)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[0].end, len(text))

    def test_boundary_preservation(self):
        text = "First sentence here. Second sentence goes here. Third sentence follows."
        chunks = self.splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        # Verify offsets
        for c in chunks:
            self.assertEqual(c.content, text[c.start:c.end])

    def test_no_punctuation_strictly_monotonic(self):
        text = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = self.splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        # Verify strictly increasing start offsets
        starts = [c.start for c in chunks]
        self.assertEqual(starts, sorted(set(starts)))

    def test_split_document(self):
        doc = Document(
            content="Alpha sentence one. Beta sentence two. Gamma sentence three.",
            metadata={"domain": "science"},
            source="doc1.txt"
        )
        chunks = self.splitter.split_document(doc)
        self.assertGreater(len(chunks), 0)
        for i, chk in enumerate(chunks):
            self.assertEqual(chk.chunk_id, i)
            self.assertEqual(chk.source, "doc1.txt")
            self.assertEqual(chk.metadata["domain"], "science")
            self.assertGreaterEqual(chk.char_offset, 0)

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            TextSplitter(chunk_size=100, chunk_overlap=100)

if __name__ == "__main__":
    unittest.main()
