import pytest
import logging
from rag_builder.storage.memory_adapter import MemoryGraphStorage
from rag_builder.storage.inspector import GraphInspector
from rag_builder.storage.base import Triplet

def test_graph_inspector_node_degree():
    storage = MemoryGraphStorage()
    inspector = GraphInspector(storage)
    
    # Create a simple graph: A -> B, A -> C, D -> A
    triplets = [
        Triplet("A", "knows", "B", {}, {}, {}),
        Triplet("A", "knows", "C", {}, {}, {}),
        Triplet("D", "knows", "A", {}, {}, {}),
    ]
    for t in triplets:
        storage.insert_triplet(t)
        
    # Node A: in=1, out=2
    degree_a = inspector.get_node_degree("A")
    assert degree_a["in_degree"] == 1
    assert degree_a["out_degree"] == 2
    assert degree_a["total_degree"] == 3
    
    # Node B: in=1, out=0
    degree_b = inspector.get_node_degree("B")
    assert degree_b["in_degree"] == 1
    assert degree_b["out_degree"] == 0
    
    # Node X: not in graph
    degree_x = inspector.get_node_degree("X")
    assert degree_x["in_degree"] == 0
    assert degree_x["out_degree"] == 0

def test_graph_inspector_relationship_distribution():
    storage = MemoryGraphStorage()
    inspector = GraphInspector(storage)
    
    triplets = [
        Triplet("A", "is_a", "Person", {}, {}, {}),
        Triplet("B", "is_a", "Person", {}, {}, {}),
        Triplet("A", "lives_in", "City", {}, {}, {}),
    ]
    for t in triplets:
        storage.insert_triplet(t)
        
    dist = inspector.get_relationship_distribution()
    assert dist["is_a"] == 2
    assert dist["lives_in"] == 1

def test_graph_inspector_connected_components():
    storage = MemoryGraphStorage()
    inspector = GraphInspector(storage)
    
    # Component 1: A-B
    storage.insert_triplet(Triplet("A", "rel", "B", {}, {}, {}))
    # Component 2: C-D
    storage.insert_triplet(Triplet("C", "rel", "D", {}, {}, {}))
    # Component 3: E (orphan)
    storage._graph.add_node("E")
    
    count = inspector.get_connected_components_count()
    assert count == 3

def test_graph_inspector_health_check():
    storage = MemoryGraphStorage()
    inspector = GraphInspector(storage)
    
    # Empty graph
    health = inspector.run_health_check()
    assert health["is_empty"] is True
    assert health["density"] == 0.0
    
    # Populated graph
    storage.insert_triplet(Triplet("A", "rel", "B", {}, {}, {}))
    health = inspector.run_health_check()
    assert health["is_empty"] is False
    assert health["node_count"] == 2
    assert health["edge_count"] == 1
    
def test_graph_inspector_print_summary(capsys):
    storage = MemoryGraphStorage()
    inspector = GraphInspector(storage)
    storage.insert_triplet(Triplet("A", "rel", "B", {}, {}, {}))
    
    inspector.print_summary()
    captured = capsys.readouterr()
    assert "Nodes: 2" in captured.out
    assert "Edges: 1" in captured.out
    assert "Backend: NetworkX" in captured.out
