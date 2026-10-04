import pytest
from unittest.mock import MagicMock
import json
from rag_builder.extraction.ollama_client import OllamaClient
from rag_builder.extraction.triplets import TripletExtractor, Triplet

@pytest.fixture
def mock_ollama_client():
    return MagicMock(spec=OllamaClient)

@pytest.fixture
def extractor(mock_ollama_client):
    return TripletExtractor(client=mock_ollama_client)

def test_extraction_success(extractor, mock_ollama_client):
    # Mock a valid JSON response from Ollama
    mock_response = '[{"subject": "Apple Inc.", "predicate": "headquartered_in", "object": "Cupertino"}]'
    mock_ollama_client.generate.return_value = mock_response
    
    text = "Apple Inc. is headquartered in Cupertino."
    triplets = extractor.extract(text)
    
    assert len(triplets) == 1
    assert triplets[0] == Triplet("Apple Inc.", "headquartered_in", "Cupertino")

def test_extraction_conversational_response(extractor, mock_ollama_client):
    # Mock a response that contains JSON but also conversational text
    mock_response = "Here are the triplets: [{\"subject\": \"Einstein\", \"predicate\": \"developed\", \"object\": \"relativity\"}] Hope this helps!"
    mock_ollama_client.generate.return_value = mock_response
    
    text = "Einstein developed relativity."
    triplets = extractor.extract(text)
    
    assert len(triplets) == 1
    assert triplets[0].subject == "Einstein"

def test_extraction_single_object_response(extractor, mock_ollama_client):
    # Mock a response that returns a single object instead of a list
    mock_response = '{"subject": "Earth", "predicate": "is_a", "object": "Planet"}'
    mock_ollama_client.generate.return_value = mock_response
    
    text = "Earth is a planet."
    triplets = extractor.extract(text)
    
    assert len(triplets) == 1
    assert triplets[0].subject == "Earth"

def test_extraction_invalid_json(extractor, mock_ollama_client):
    # Mock a response that is not valid JSON
    mock_ollama_client.generate.return_value = "This is not JSON"
    
    text = "Invalid response test."
    triplets = extractor.extract(text)
    
    assert triplets == []

def test_extraction_empty_text(extractor, mock_ollama_client):
    triplets = extractor.extract("")
    assert triplets == []
    mock_ollama_client.generate.assert_not_called()

def test_extraction_no_response(extractor, mock_ollama_client):
    mock_ollama_client.generate.return_value = None
    
    text = "No response test."
    triplets = extractor.extract(text)
    
    assert triplets == []
