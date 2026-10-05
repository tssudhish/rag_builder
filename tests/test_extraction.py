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


def test_ollama_client_url_normalization():
    # Test trailing slash stripping and /api/generate stripping
    client1 = OllamaClient("http://localhost:11434/")
    assert client1.base_url == "http://localhost:11434"

    client2 = OllamaClient("http://localhost:11434/api/generate")
    assert client2.base_url == "http://localhost:11434"

    client3 = OllamaClient("http://localhost:11434/api/")
    assert client3.base_url == "http://localhost:11434"


def test_ollama_client_model_resolution():
    client = OllamaClient()
    assert client.resolve_model("sciphi-triplex") == "sciphi/triplex:latest"
    assert client.resolve_model("sciphi/triplex") == "sciphi/triplex:latest"
    assert client.resolve_model("custom-model") == "custom-model"


def test_ollama_client_generate_success(monkeypatch):
    client = OllamaClient()
    
    class MockResponse:
        status_code = 200
        def json(self):
            return {"response": '{"subject": "A", "predicate": "B", "object": "C"}'}
        def raise_for_status(self):
            pass

    monkeypatch.setattr("requests.post", lambda url, json, timeout: MockResponse())
    
    result = client.generate("sciphi-triplex", "test prompt")
    assert result == '{"subject": "A", "predicate": "B", "object": "C"}'


def test_ollama_client_404_fallback(monkeypatch):
    client = OllamaClient()

    # Simulate /api/tags returning installed models
    class MockTagsResponse:
        status_code = 200
        def json(self):
            return {"models": [{"name": "sciphi/triplex:latest"}]}

    monkeypatch.setattr("requests.get", lambda url, timeout: MockTagsResponse())

    call_count = {"count": 0}
    def mock_post(url, json, timeout):
        call_count["count"] += 1
        if json.get("model") == "unknown-model":
            class Mock404:
                status_code = 404
                text = '{"error": "model not found"}'
            return Mock404()
        class Mock200:
            status_code = 200
            def json(self):
                return {"response": "fallback success"}
        return Mock200()

    monkeypatch.setattr("requests.post", mock_post)

    # unknown-model should fail and return None
    res = client.generate("unknown-model", "test")
    assert res is None

