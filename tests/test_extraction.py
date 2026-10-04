import unittest
from unittest.mock import MagicMock
from rag_builder.extraction import OllamaClient, Triplet, TripletExtractor

class TestTripletExtraction(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=OllamaClient)
        self.extractor = TripletExtractor(self.mock_client)

    def test_extract_empty_text(self):
        """Verify that empty or whitespace text returns an empty list."""
        self.assertEqual(self.extractor.extract(""), [])
        self.assertEqual(self.extractor.extract("   "), [])
        self.mock_client.generate.assert_not_called()

    def test_extract_successful_list(self):
        """Verify extraction from a standard JSON list response."""
        mock_response = '[{"subject": "Apple", "predicate": "is", "object": "company"}]'
        self.mock_client.generate.return_value = mock_response
        
        results = self.extractor.extract("Apple is a company.")
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Triplet)
        self.assertEqual(results[0].subject, "Apple")
        self.assertEqual(results[0].predicate, "is")
        self.assertEqual(results[0].obj, "company")

    def test_extract_conversational_response(self):
        """Verify that JSON is extracted from within conversational text."""
        mock_response = 'Sure! Here are the triplets: [{"subject": "Einstein", "predicate": "born_in", "object": "Germany"}] Hope this helps!'
        self.mock_client.generate.return_value = mock_response
        
        results = self.extractor.extract("Einstein was born in Germany.")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].subject, "Einstein")

    def test_extract_single_object_response(self):
        """Verify that a single JSON object is handled as a single triplet."""
        mock_response = '{"subject": "Python", "predicate": "is", "object": "language"}'
        self.mock_client.generate.return_value = mock_response
        
        results = self.extractor.extract("Python is a language.")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].subject, "Python")

    def test_extract_malformed_json(self):
        """Verify that malformed JSON returns an empty list."""
        self.mock_client.generate.return_value = 'This is not JSON'
        results = self.extractor.extract("Some text")
        self.assertEqual(results, [])

    def test_extract_client_error(self):
        """Verify that None response from client returns an empty list."""
        self.mock_client.generate.return_value = None
        results = self.extractor.extract("Some text")
        self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main()
