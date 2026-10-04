import pytest
from fastapi.testclient import TestClient
from rag_builder.api.main import app
from rag_builder.api.state import app_state
from rag_builder.storage.memory import MemoryGraphStorage
from rag_builder.storage.base import Triplet

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_graph():
    # Reset storage for each test
    storage = MemoryGraphStorage()
    app_state.set_storage(storage)
    yield storage

def test_graph_visualization_empty():
    response = client.get("/api/graph/nodes-and-edges")
    assert response.status_code == 200
    data = response.json()
    assert data == {"nodes": [], "edges": []}

def test_graph_visualization_with_data():
    storage = app_state.get_storage()
    storage.insert_triplet(Triplet(subject="A", predicate="is", object="B"))
    storage.insert_triplet(Triplet(subject="B", predicate="is", object="C"))
    
    response = client.get("/api/graph/nodes-and-edges")
    assert response.status_code == 200
    data = response.json()
    
    # Expect 3 nodes (A, B, C) and 2 edges
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2

def test_graph_visualization_limit():
    storage = app_state.get_storage()
    # Insert 150 triplets
    for i in range(150):
        storage.insert_triplet(Triplet(subject=f"S{i}", predicate="rel", object=f"O{i}"))
    
    # Test default limit (100)
    response = client.get("/api/graph/nodes-and-edges")
    data = response.json()
    assert len(data["edges"]) == 100
    
    # Test custom limit (50)
    response = client.get("/api/graph/nodes-and-edges?limit=50")
    data = response.json()
    assert len(data["edges"]) == 50
    
    # Test high limit (200)
    response = client.get("/api/graph/nodes-and-edges?limit=200")
    data = response.json()
    assert len(data["edges"]) == 150 # Since only 150 exist
