import pytest
import logging
from rag_builder.storage import Triplet, MemoryGraphStorage, Neo4jGraphStorage
from rag_builder.storage.inspector import GraphInspector

# Setup logging for tests to see output
logging.basicConfig(level=logging.INFO)

def test_memory_storage_basic():
    storage = MemoryGraphStorage()
    
    # Create a small chain: A -> (works_at) -> B -> (located_in) -> C
    t1 = Triplet("Alice", "works_at", "TechCorp", subject_properties={"age": 30}, object_properties={"industry": "AI"})
    t2 = Triplet("TechCorp", "located_in", "New York", predicate_properties={"since": 2010})
    
    storage.insert_triplet(t1)
    storage.insert_triplet(t2)
    
    stats = storage.get_stats()
    assert stats.node_count == 3
    assert stats.edge_count == 2
    assert stats.unique_predicates == 2
    
    # Test query neighbors depth 1
    neighbors_1 = list(storage.query_neighbors("Alice", depth=1))
    assert ("Alice", "works_at", "TechCorp") in neighbors_1
    assert len(neighbors_1) == 1
    
    # Test query neighbors depth 2
    neighbors_2 = list(storage.query_neighbors("Alice", depth=2))
    assert ("Alice", "works_at", "TechCorp") in neighbors_2
    assert ("TechCorp", "located_in", "New York") in neighbors_2
    assert len(neighbors_2) == 2

def test_memory_storage_diagnostics():
    storage = MemoryGraphStorage()
    # Component 1: A -> B
    storage.insert_triplet(Triplet("A", "rel", "B"))
    # Component 2: C (orphan) - To create an orphan in MemoryGraphStorage, 
    # we can't use insert_triplet because it creates an edge.
    # But we can access the internal graph for test setup.
    storage._graph.add_node("C")
    
    assert storage.get_connected_components() == 2
    assert "C" in storage.get_orphan_nodes()
    assert len(storage.get_orphan_nodes()) == 1
    
    degree_a = storage.get_node_degree("A")
    assert degree_a["out_degree"] == 1
    assert degree_a["in_degree"] == 0
    
    degree_b = storage.get_node_degree("B")
    assert degree_b["in_degree"] == 1
    assert degree_b["out_degree"] == 0
    
    degree_c = storage.get_node_degree("C")
    assert degree_c["in_degree"] == 0
    assert degree_c["out_degree"] == 0

def test_memory_storage_direction():

    storage = MemoryGraphStorage()
    storage.insert_triplet(Triplet("A", "rel", "B"))
    
    # Outgoing
    assert list(storage.query_neighbors("A", direction="out")) == [("A", "rel", "B")]
    # Incoming
    assert list(storage.query_neighbors("B", direction="in")) == [("A", "rel", "B")]
    # Both
    assert list(storage.query_neighbors("A", direction="both")) == [("A", "rel", "B")]
    assert list(storage.query_neighbors("B", direction="both")) == [("A", "rel", "B")]

@pytest.mark.skip(reason="Neo4j requires a running server")
def test_neo4j_storage_basic():
    # This test is skipped by default but provided for integration testing
    storage = Neo4jGraphStorage(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    t1 = Triplet("Alice", "works_at", "TechCorp")
    t2 = Triplet("TechCorp", "located_in", "New York")
    
    storage.insert_triplet(t1)
    storage.insert_triplet(t2)
    
    stats = storage.get_stats()
    assert stats.node_count >= 2
    
    neighbors = list(storage.query_neighbors("Alice", depth=2))
    assert any(t == ("Alice", "works_at", "TechCorp") for t in neighbors)
    
    storage.close()

def test_neo4j_sanitization():
    # We can test the internal sanitization method directly without a DB connection
    storage = Neo4jGraphStorage.__new__(Neo4jGraphStorage) # Bypass __init__
    
    # Test safe predicate
    assert storage._sanitize_predicate("works_at") == "works_at"
    # Test unsafe predicate (Cypher injection attempt)
    assert storage._sanitize_predicate("works_at)-[r]->(m") == "works_at___r____m"
    # Test special characters
    assert storage._sanitize_predicate("is-a-member-of!") == "is_a_member_of_"
