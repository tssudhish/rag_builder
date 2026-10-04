import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from rag_builder.cli import build_graph, query_graph, main
from rag_builder.storage.memory_adapter import MemoryGraphStorage
from rag_builder.extraction.triplets import Triplet

def test_e2e_pipeline(tmp_path):
    """
    Tests the end-to-end pipeline with mocked LLM calls.
    Document -> Chunks -> Triplets -> Graph -> Query -> Answer.
    """
    # 1. Setup sample data
    sample_text = "Apple Inc. is a technology company based in Cupertino. Steve Jobs co-founded Apple Inc."
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text(sample_text)
    
    # Mock the TripletExtractor.extract method
    with patch('rag_builder.cli.TripletExtractor.extract') as mock_extract:
        mock_extract.return_value = [
            Triplet(subject="Apple Inc.", predicate="based_in", obj="Cupertino"),
            Triplet(subject="Steve Jobs", predicate="co-founded", obj="Apple Inc.")
        ]
        
        # 2. Build Graph
        try:
            storage = build_graph(str(sample_file), model="sciphi-triplex")
        except Exception as e:
            pytest.fail(f"Build graph failed: {e}")

    # 3. Verify Graph Content
    stats = storage.get_stats()
    assert stats.node_count > 0
    assert stats.edge_count > 0

    # 4. Query the Graph
    query = "Where is Apple Inc. based?"
    
    with patch('rag_builder.cli.ResponseGenerator.generate') as mock_generate:
        from rag_builder.rag.generator import GeneratedResponse
        mock_generate.return_value = GeneratedResponse(
            answer="Apple Inc. is based in Cupertino.",
            citations=[("Apple Inc.", "based_in", "Cupertino")],
            context_used="Apple Inc. --(based_in)--> Cupertino"
        )
        
        try:
            response = query_graph(storage, query, model="gemma2")
        except Exception as e:
            pytest.fail(f"Query graph failed: {e}")

    # 5. Validate Response
    assert response.answer == "Apple Inc. is based in Cupertino."
    assert len(response.citations) > 0

def test_graph_persistence(tmp_path):
    """
    Tests saving and loading the graph using MemoryGraphStorage.
    """
    storage = MemoryGraphStorage()
    from rag_builder.storage.base import Triplet
    storage.insert_triplet(Triplet(subject="Entity A", predicate="rel", object="Entity B"))
    
    save_path = tmp_path / "graph.json"
    storage.export_json(str(save_path))
    
    assert save_path.exists()
    
    # Load into a new storage instance
    new_storage = MemoryGraphStorage()
    new_storage.import_json(str(save_path))
    
    stats = new_storage.get_stats()
    assert stats.node_count == 2
    assert stats.edge_count == 1

def test_cli_execution(tmp_path):
    """
    Tests CLI entry points using sys.argv mocking.
    """
    import sys
    
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Test content")
    graph_file = tmp_path / "graph.json"
    
    # Mock build command
    with patch('sys.argv', ["rag-builder", "build", "--file", str(sample_file), "--save", str(graph_file)]), \
         patch('rag_builder.cli.TripletExtractor.extract', return_value=[]):
        main()
    
    assert graph_file.exists()

    # Mock query command with load
    with patch('sys.argv', ["rag-builder", "query", "--query", "test?", "--load", str(graph_file)]), \
         patch('rag_builder.cli.ResponseGenerator.generate') as mock_gen:
        
        from rag_builder.rag.generator import GeneratedResponse
        mock_gen.return_value = GeneratedResponse(answer="Test answer", citations=[], context_used="")
        main()
        
        mock_gen.assert_called_once()

def test_cli_help():
    """
    Verifies that the CLI module can be imported and main can be called with --help.
    """
    import sys
    with pytest.raises(SystemExit) as e:
        with patch('sys.argv', ["rag-builder", "--help"]):
            main()
    assert e.value.code == 0
