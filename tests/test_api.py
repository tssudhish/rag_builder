import pytest
from fastapi.testclient import TestClient
from rag_builder.api.main import app
import time

client = TestClient(app)

def test_upload_valid_document():
    # Test valid .txt file
    file_content = b"Hello world, this is a test document."
    response = client.post(
        "/api/documents",
        files={"file": ("test.txt", file_content, "text/plain")}
    )
    assert response.status_code == 202
    data = response.json()
    assert "document_id" in data
    doc_id = data["document_id"]

    # Verify initial status
    status_res = client.get(f"/api/documents/status/{doc_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] in ["pending", "processing", "completed"]

def test_upload_invalid_extension():
    # Test invalid .exe file
    file_content = b"binary data"
    response = client.post(
        "/api/documents",
        files={"file": ("malicious.exe", file_content, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]

def test_document_lifecycle():
    # Upload a document and wait for it to complete
    file_content = b"Lifecycle test content"
    response = client.post(
        "/api/documents",
        files={"file": ("lifecycle.md", file_content, "text/markdown")}
    )
    assert response.status_code == 202
    doc_id = response.json()["document_id"]

    # Poll for completion (with timeout)
    max_retries = 10
    completed = False
    for _ in range(max_retries):
        status_res = client.get(f"/api/documents/status/{doc_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        if status_data["status"] == "completed":
            completed = True
            break
        time.sleep(1)
    
    assert completed, "Document processing did not complete in time"
    assert client.get(f"/api/documents/status/{doc_id}").json()["progress"] == 1.0

def test_get_graph_stats():
    # Initial stats
    response = client.get("/api/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert "node_count" in data
    assert "edge_count" in data

def test_get_nonexistent_document():
    response = client.get("/api/documents/status/non-existent-id")
    assert response.status_code == 404
