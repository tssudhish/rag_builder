import pytest
from rag_builder.storage.base import Triplet
from rag_builder.storage.memory_adapter import MemoryGraphStorage
from rag_builder.agent_tools.langchain_tools import KnowledgeGraphToolSet

@pytest.fixture
def sample_storage():
    storage = MemoryGraphStorage()
    triplets = [
        Triplet("Python", "is_a", "Programming Language"),
        Triplet("Python", "created_by", "Guido van Rossum"),
        Triplet("Guido van Rossum", "born_in", "Netherlands"),
        Triplet("LangChain", "framework_for", "LLMs"),
        Triplet("LangChain", "uses", "Python"),
    ]
    for t in triplets:
        storage.insert_triplet(t)
    return storage

@pytest.fixture
def tool_set(sample_storage):
    return KnowledgeGraphToolSet(sample_storage)

def test_tool_set_initialization(tool_set):
    tools = tool_set.get_tools()
    assert len(tools) == 3
    tool_names = [t.name for t in tools]
    assert "query_graph" in tool_names
    assert "get_entity_neighbors" in tool_names
    assert "search_triplets" in tool_names

def test_query_graph_tool(tool_set):
    # Test matching subject
    res = tool_set.query_graph("Python")
    assert "Python --is_a--> Programming Language" in res
    assert "Python --created_by--> Guido van Rossum" in res
    assert "LangChain --uses--> Python" in res

    # Test matching object
    res = tool_set.query_graph("Netherlands")
    assert "Guido van Rossum --born_in--> Netherlands" in res

    # Test no match
    res = tool_set.query_graph("JavaScript")
    assert "No information found for 'JavaScript'." in res

def test_get_entity_neighbors_tool(tool_set):
    # Test neighbors of Python (depth 1)
    res = tool_set.get_entity_neighbors("Python", depth=1)
    assert "Python --is_a--> Programming Language" in res
    assert "Python --created_by--> Guido van Rossum" in res
    assert "LangChain --uses--> Python" in res

    # Test neighbors of Guido (depth 1)
    res = tool_set.get_entity_neighbors("Guido van Rossum", depth=1)
    assert "Python --created_by--> Guido van Rossum" in res
    assert "Guido van Rossum --born_in--> Netherlands" in res

    # Test depth 2 (Python -> Guido -> Netherlands)
    res = tool_set.get_entity_neighbors("Python", depth=2)
    assert "Guido van Rossum --born_in--> Netherlands" in res

    # Test non-existent entity
    res = tool_set.get_entity_neighbors("Mars", depth=1)
    assert "No neighbors found for entity 'Mars'." in res

def test_search_triplets_tool(tool_set):
    # Test existing predicate
    res = tool_set.search_triplets("is_a")
    assert "Python --is_a--> Programming Language" in res

    # Test another predicate
    res = tool_set.search_triplets("born_in")
    assert "Guido van Rossum --born_in--> Netherlands" in res

    # Test non-existent predicate
    res = tool_set.search_triplets("works_at")
    assert "No triplets found with predicate 'works_at'." in res

def test_langchain_tool_schemas(tool_set):
    tools = tool_set.get_tools()
    
    # Find specific tools
    query_tool = next(t for t in tools if t.name == "query_graph")
    neighbors_tool = next(t for t in tools if t.name == "get_entity_neighbors")
    search_tool = next(t for t in tools if t.name == "search_triplets")
    
    # Verify schemas are present
    assert query_tool.args_schema is not None
    assert neighbors_tool.args_schema is not None
    assert search_tool.args_schema is not None
    
    # Verify specific schema fields
    assert "query" in query_tool.args_schema.model_fields
    assert "entity_name" in neighbors_tool.args_schema.model_fields
    assert "predicate" in search_tool.args_schema.model_fields

def test_tool_execution_via_langchain_interface(tool_set):
    tools = tool_set.get_tools()
    query_tool = next(t for t in tools if t.name == "query_graph")
    
    # Execute via the tool's .run() method (standard LangChain interface)
    result = query_tool.run({"query": "Python"})
    assert "Python --is_a--> Programming Language" in result

    # Execute via the .invoke() method
    result_invoke = query_tool.invoke({"query": "Python"})
    assert result_invoke == result

def test_tool_json_schema(tool_set):
    tools = tool_set.get_tools()
    query_tool = next(t for t in tools if t.name == "query_graph")
    schema = query_tool.get_json_schema()
    
    assert schema["name"] == "query_graph"
    assert "parameters" in schema
    assert schema["parameters"]["properties"]["query"]["type"] == "string"

def test_tool_validation_errors(tool_set):
    tools = tool_set.get_tools()
    query_tool = next(t for t in tools if t.name == "query_graph")
    
    # Test invalid input type (dict with wrong key)
    result = query_tool.run({"wrong_key": "Python"})
    assert "Input validation error" in result
    
    # Test missing required field
    result_missing = query_tool.run({})
    assert "Input validation error" in result_missing
