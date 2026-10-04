import pytest
from rag_builder.storage.models import Entity, Triplet
from rag_builder.storage.vector_index import VectorIndex
from rag_builder.storage.pipeline import IngestionPipeline, InMemoryGraphStore

def test_pipeline_ingestion():
    # Setup
    graph_store = InMemoryGraphStore()
    vector_index = VectorIndex(dimension=128)
    pipeline = IngestionPipeline(graph_store, vector_index)

    # Sample triplets
    e1 = Entity(name="Python", label="Language")
    e2 = Entity(name="Guido van Rossum", label="Person")
    e3 = Entity(name="Netherlands", label="Country")

    triplets = [
        Triplet(subject=e2, predicate="created", object=e1),
        Triplet(subject=e2, predicate="born_in", object=e3),
        # Duplicate triplet
        Triplet(subject=e2, predicate="created", object=e1),
        # Triplet with existing node
        Triplet(subject=e1, predicate="developed_in", object=e3),
    ]

    stats = pipeline.ingest_batch(triplets)

    # Verification
    assert stats["nodes_added"] == 3
    assert stats["edges_added"] == 3
    assert stats["embeddings_created"] == 3
    
    assert "Python" in graph_store.nodes
    assert "Guido van Rossum" in graph_store.nodes
    assert "Netherlands" in graph_store.nodes
    
    assert vector_index.get_embedding("Python") is not None
    assert len(vector_index.get_embedding("Python")) == 128

def test_deduplication():
    graph_store = InMemoryGraphStore()
    vector_index = VectorIndex(dimension=128)
    pipeline = IngestionPipeline(graph_store, vector_index)

    e1 = Entity(name="A", label="L")
    e2 = Entity(name="B", label="L")
    
    # Same triplet multiple times
    triplets = [
        Triplet(subject=e1, predicate="rel", object=e2),
        Triplet(subject=e1, predicate="rel", object=e2),
        Triplet(subject=e1, predicate="rel", object=e2),
    ]

    stats = pipeline.ingest_batch(triplets)
    assert stats["nodes_added"] == 2
    assert stats["edges_added"] == 1

def test_vector_search():
    vector_index = VectorIndex(dimension=4)
    e1 = Entity(name="Apple", label="Fruit")
    e2 = Entity(name="Banana", label="Fruit")
    
    # Manually add embeddings for controlled testing
    vector_index.add_embedding(e1, [1.0, 0.0, 0.0, 0.0])
    vector_index.add_embedding(e2, [0.0, 1.0, 0.0, 0.0])
    
    # Search for something close to Apple
    results = vector_index.search([0.9, 0.1, 0.0, 0.0], top_k=1)
    assert results[0][0] == "Apple"
    
    # Search for something close to Banana
    results = vector_index.search([0.1, 0.9, 0.0, 0.0], top_k=1)
    assert results[0][0] == "Banana"

def test_embedding_dimension_mismatch():
    vector_index = VectorIndex(dimension=128)
    e1 = Entity(name="Test", label="L")
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        vector_index.add_embedding(e1, [0.1] * 64)

def test_empty_batch_ingestion():
    graph_store = InMemoryGraphStore()
    vector_index = VectorIndex(dimension=128)
    pipeline = IngestionPipeline(graph_store, vector_index)
    
    stats = pipeline.ingest_batch([])
    assert stats == {"nodes_added": 0, "edges_added": 0, "embeddings_created": 0}

def test_cross_batch_deduplication():
    graph_store = InMemoryGraphStore()
    vector_index = VectorIndex(dimension=128)
    pipeline = IngestionPipeline(graph_store, vector_index)
    
    e1 = Entity(name="Node1", label="L")
    e2 = Entity(name="Node2", label="L")
    
    # Batch 1
    batch1 = [Triplet(subject=e1, predicate="rel", object=e2)]
    stats1 = pipeline.ingest_batch(batch1)
    assert stats1["nodes_added"] == 2
    assert stats1["embeddings_created"] == 2
    
    # Batch 2 - reuse nodes but add new edge
    e3 = Entity(name="Node3", label="L")
    batch2 = [Triplet(subject=e1, predicate="rel2", object=e3)]
    stats2 = pipeline.ingest_batch(batch2)
    
    # Node1 already exists, so nodes_added should only count Node3
    assert stats2["nodes_added"] == 1
    # Node1 already has embedding, so embeddings_created should only count Node3
    assert stats2["embeddings_created"] == 1
    assert stats2["edges_added"] == 1

def test_graph_store_get_all_edges():
    graph_store = InMemoryGraphStore()
    e1 = Entity(name="A", label="L")
    e2 = Entity(name="B", label="L")
    t1 = Triplet(subject=e1, predicate="rel", object=e2, properties={"weight": 1.0})
    
    graph_store.upsert_node(e1)
    graph_store.upsert_node(e2)
    graph_store.upsert_edge(t1)
    
    edges = graph_store.get_all_edges()
    assert len(edges) == 1
    assert edges[0].subject.name == "A"
    assert edges[0].predicate == "rel"
    assert edges[0].object.name == "B"
    assert edges[0].properties["weight"] == 1.0

