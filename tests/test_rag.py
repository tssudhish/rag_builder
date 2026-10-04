import pytest
from unittest.mock import MagicMock, patch
from rag_builder.rag import (
    ResponseGenerator, 
    RAGEvaluator, 
    SubGraphResult, 
    GeneratedResponse,
    RAGEvaluationResult
)
import requests

@pytest.fixture
def mock_subgraph_results():
    return [
        SubGraphResult(
            target_node="Alice",
            triplets=[("Alice", "works_for", "Acme Corp"), ("Alice", "lives_in", "New York")],
            context_string="Knowledge Graph Context for 'Alice':\n- Alice --(works_for)--> Acme Corp\n- Alice --(lives_in)--> New York"
        ),
        SubGraphResult(
            target_node="Acme Corp",
            triplets=[("Acme Corp", "located_in", "USA")],
            context_string="Knowledge Graph Context for 'Acme Corp':\n- Acme Corp --(located_in)--> USA"
        )
    ]

@pytest.fixture
def generator():
    return ResponseGenerator(model="gemma2")

@pytest.fixture
def evaluator():
    return RAGEvaluator(model="gemma2")

def test_response_generation_index_citations(generator, mock_subgraph_results):
    """Test that the generator correctly handles index-based citations [T0], [T1]."""
    question = "Where does Alice work and where is that company located?"
    
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # The generator now expects citations as index IDs like [T0], [T1]
        mock_response.json.return_value = {
            "response": '{"answer": "Alice works for Acme Corp, which is located in the USA.", "citations": ["[T0]", "[T2]"]}'
        }
        mock_post.return_value = mock_response

        res = generator.generate(question, mock_subgraph_results)
        
        assert isinstance(res, GeneratedResponse)
        assert "Alice works for Acme Corp" in res.answer
        # [T0] should be ("Alice", "works_for", "Acme Corp")
        # [T1] should be ("Alice", "lives_in", "New York")
        # [T2] should be ("Acme Corp", "located_in", "USA")
        assert len(res.citations) == 2
        assert ("Alice", "works_for", "Acme Corp") in res.citations
        assert ("Acme Corp", "located_in", "USA") in res.citations

def test_response_generation_error_handling(generator, mock_subgraph_results):
    """Test generator resilience when LLM returns invalid JSON or network error."""
    question = "Where does Alice work?"
    
    with patch('requests.post') as mock_post:
        # Case 1: Network Error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        res = generator.generate(question, mock_subgraph_results)
        assert "An error occurred" in res.answer
        assert len(res.citations) == 0

        # Case 2: Invalid JSON in response field
        mock_post.side_effect = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Not JSON at all"}
        mock_post.return_value = mock_response
        res = generator.generate(question, mock_subgraph_results)
        assert "An error occurred" in res.answer

def test_evaluator_resilience_and_retries(evaluator):
    """Test that the evaluator retries on failure and handles system failure correctly."""
    question = "What is Alice's job?"
    answer = "Alice is a software engineer."
    context = "Alice --(works_for)--> Acme Corp"
    
    with patch('requests.post') as mock_post:
        # Mock failure for all attempts
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = evaluator.evaluate(question, answer, context)
        
        # Should return 0.0 and a specific "System failure" reasoning
        assert result.metrics[0].score == 0.0
        assert "System failure" in result.metrics[0].reasoning
        # Check that it actually retried 3 times
        assert mock_post.call_count == 3

def test_evaluator_successful_recovery(evaluator):
    """Test that evaluator recovers if a retry succeeds."""
    question = "What is Alice's job?"
    answer = "Alice is a software engineer."
    context = "Alice --(works_for)--> Acme Corp"
    
    with patch('requests.post') as mock_post:
        # Fail twice, succeed on third
        fail_response = requests.exceptions.HTTPError("500 Internal Server Error")
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"response": '{"score": 1.0, "reasoning": "Perfect"}'}
        
        mock_post.side_effect = [
            fail_response,
            fail_response,
            success_response
        ]
        
        result = evaluator.evaluate(question, answer, context)
        assert result.metrics[0].score == 1.0
        assert "Perfect" in result.metrics[0].reasoning
        assert mock_post.call_count == 3

def test_evaluator_faithfulness_standard(evaluator):
    """Test standard faithfulness check."""
    question = "What is Alice's job?"
    answer = "Alice is a software engineer."
    context = "Alice --(works_for)--> Acme Corp"
    
    with patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"score": 0.0, "reasoning": "Not in context"}'
        }
        mock_post.return_value = mock_response
        
        result = evaluator.evaluate(question, answer, context)
        assert result.metrics[0].score == 0.0
        assert "Not in context" in result.metrics[0].reasoning
