import pytest
from rag_builder.storage.memory_adapter import MemoryGraphStorage
from rag_builder.storage.base import Triplet
from rag_builder.rag.retriever import GraphRetriever, SubGraphResult

@pytest.fixture
def sample_graph():
    storage = MemoryGraphStorage()
    # Create a small network: A -> B -> C, A -> D, B -> E
    triplets = [
        Triplet("Apple", "is_a", "Fruit"),
        Triplet("Fruit", "has_property", "Seed"),
        Triplet("Apple", "originates_from", "Central Asia"),
        Triplet("Central Asia", "is_a", "Region"),
        Triplet("Fruit", "is_a", "PlantPart"),
    ]
    for t in triplets:
        storage.insert_triplet(t)
    return storage

def test_k_hop_retrieval_1_hop(sample_graph):
    retriever = GraphRetriever(sample_graph)
    # Apple's 1-hop neighbors: (Apple, is_a, Fruit), (Apple, originates_from, Central Asia)
    result = retriever.retrieve_k_hop_neighborhood("Apple", k=1)
    
    assert len(result.triplets) >= 2
    # Verify specific relationships
    triplet_strings = [f"{s} --({p})--> {o}" for s, p, o in result.triplets]
    assert "Apple --(is_a)--> Fruit" in triplet_strings
    assert "Apple --(originates_from)--> Central Asia" in triplet_strings
    assert "Fruit --(has_property)--> Seed" not in triplet_strings

def test_k_hop_retrieval_2_hops(sample_graph):
    retriever = GraphRetriever(sample_graph)
    # Apple's 2-hop neighbors should include seeds and region info
    result = retriever.retrieve_k_hop_neighborhood("Apple", k=2)
    
    triplet_strings = [f"{s} --({p})--> {o}" for s, p, o in result.triplets]
    assert "Apple --(is_a)--> Fruit" in triplet_strings
    assert "Fruit --(has_property)--> Seed" in triplet_strings
    assert "Central Asia --(is_a)--> Region" in triplet_strings

def test_k_hop_retrieval_depth_limit(sample_graph):
    retriever = GraphRetriever(sample_graph)
    # 0 hops should return nothing (or just the node, depending on implementation, 
    # but our query_neighbors returns triplets, so 0 depth usually means no edges)
    result = retriever.retrieve_k_hop_neighborhood("Apple", k=0)
    assert len(result.triplets) == 0

def test_prompt_context_formatting(sample_graph):
    retriever = GraphRetriever(sample_graph)
    result = retriever.retrieve_k_hop_neighborhood("Apple", k=1)
    
    context = result.context_string
    assert "Knowledge Graph Context for 'Apple':" in context
    assert "Apple --(is_a)--> Fruit" in context
    assert "Apple --(originates_from)--> Central Asia" in context

def test_non_existent_node(sample_graph):
    retriever = GraphRetriever(sample_graph)
    result = retriever.retrieve_k_hop_neighborhood("NonExistent", k=1)
    
    assert len(result.triplets) == 0
    assert "No structural context found for entity: NonExistent" in result.context_string

def test_relevance_scoring(sample_graph):
    def mock_scoring_fn(node, triplets):
        return float(len(triplets)) * 0.1
    
    retriever = GraphRetriever(sample_graph)
    result = retriever.retrieve_k_hop_neighborhood("Apple", k=1, scoring_fn=mock_scoring_fn)
    
    # Apple has 2 triplets in 1-hop
    assert result.relevance_score == pytest.approx(0.2)

def test_direction_filter(sample_graph):
    # To test direction, we need a clear direction: A -> B
    storage = MemoryGraphStorage()
    storage.insert_triplet(Triplet("A", "likes", "B"))
    retriever = GraphRetriever(storage)
    
    # Outgoing from A
    res_out = retriever.retrieve_k_hop_neighborhood("A", k=1, direction="out")
    assert ("A", "likes", "B") in res_out.triplets
    
    # Incoming to A (should be empty)
    res_in = retriever.retrieve_k_hop_neighborhood("A", k=1, direction="in")
    assert len(res_in.triplets) == 0
    
    # Incoming to B (should find A -> B)
    res_in_b = retriever.retrieve_k_hop_neighborhood("B", k=1, direction="in")
    assert ("A", "likes", "B") in res_in_b.triplets
