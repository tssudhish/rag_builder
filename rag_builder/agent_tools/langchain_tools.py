from typing import Any, List, Dict, Optional, Protocol, Union
from pydantic import BaseModel, Field, ValidationError

class BaseToolProtocol(Protocol):
    name: str
    description: str
    args_schema: Any
    def run(self, tool_input: Any) -> str: ...
    def invoke(self, tool_input: Any) -> str: ...
    def get_json_schema(self) -> Dict[str, Any]: ...

class QueryGraphInput(BaseModel):
    """Input for querying the knowledge graph using a general query."""
    query: str = Field(description="The search term or entity to look for in the graph")

class EntityNeighborsInput(BaseModel):
    """Input for getting neighbors of a specific entity."""
    entity_name: str = Field(description="The exact name of the entity to find neighbors for")
    depth: int = Field(default=1, description="Number of hops to traverse from the entity")
    direction: str = Field(default="both", description="Direction of traversal: 'in', 'out', or 'both'")

class SearchTripletsInput(BaseModel):
    """Input for searching triplets based on a predicate."""
    predicate: str = Field(description="The relationship type (predicate) to filter triplets by")

class GenericTool:
    """
    A lightweight tool implementation that mimics LangChain's BaseTool
    to avoid heavy dependencies in the core logic while remaining compatible
    with the expected interface.
    """
    def __init__(self, name: str, func: Any, description: str, args_schema: Any):
        self.name = name
        self._func = func
        self.description = description
        self.args_schema = args_schema

    def run(self, tool_input: Any) -> str:
        try:
            if isinstance(tool_input, dict):
                # Validate input against the Pydantic schema
                validated_input = self.args_schema.model_validate(tool_input)
                # Pass validated fields as keyword arguments to the function
                return self._func(**validated_input.model_dump())
            
            # If input is not a dict, try to instantiate the schema with it
            validated_input = self.args_schema(**tool_input) if not isinstance(tool_input, (str, int, float, bool)) else \
                               self.args_schema(query=tool_input) if self.name == "query_graph" else \
                               self.args_schema(entity_name=tool_input) if self.name == "get_entity_neighbors" else \
                               self.args_schema(predicate=tool_input)
            
            return self._func(**validated_input.model_dump())
        except ValidationError as e:
            return f"Input validation error for tool '{self.name}': {str(e)}"
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"

    def invoke(self, tool_input: Any) -> str:
        """Alias for run() for LangChain Runnable compatibility."""
        return self.run(tool_input)

    def get_json_schema(self) -> Dict[str, Any]:
        """Returns the OpenAI/function-calling JSON schema using the args_schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema()
        }

class KnowledgeGraphToolSet:
    """
    A collection of tools for interacting with the Knowledge Graph.
    Compatible with LangChain / function calling patterns.
    """
    def __init__(self, storage: 'BaseGraphStorage'):
        self.storage = storage

    def query_graph(self, query: str) -> str:
        """
        Search the knowledge graph for a specific entity or keyword.
        Returns a list of triplets involving the query term.
        """
        try:
            triplets = self.storage.get_all_triplets()
            results = [
                f"{t.subject} --{t.predicate}--> {t.object}" 
                for t in triplets 
                if query.lower() in t.subject.lower() or query.lower() in t.object.lower()
            ]
            if not results:
                return f"No information found for '{query}'."
            return "\n".join(results)
        except Exception as e:
            return f"Error querying the graph for '{query}': {str(e)}"

    def get_entity_neighbors(self, entity_name: str, depth: int = 1, direction: str = "both") -> str:
        """
        Retrieve the neighborhood of a specific entity.
        Useful for exploring relationships around a known concept.
        """
        try:
            neighbors = list(self.storage.query_neighbors(node_id=entity_name, depth=depth, direction=direction))
            if not neighbors:
                return f"No neighbors found for entity '{entity_name}'."
            
            results = [f"{s} --{p}--> {o}" for s, p, o in neighbors]
            return "\n".join(results)
        except Exception as e:
            return f"Error retrieving neighbors for '{entity_name}': {str(e)}"

    def search_triplets(self, predicate: str) -> str:
        """
        Find all triplets that share a specific relationship type (predicate).
        Useful for analyzing specific types of connections across the graph.
        """
        try:
            triplets = self.storage.get_all_triplets()
            results = [
                f"{t.subject} --{t.predicate}--> {t.object}" 
                for t in triplets 
                if t.predicate.lower() == predicate.lower()
            ]
            if not results:
                return f"No triplets found with predicate '{predicate}'."
            return "\n".join(results)
        except Exception as e:
            return f"Error searching triplets with predicate '{predicate}': {str(e)}"

    def get_tools(self) -> List[BaseToolProtocol]:
        """
        Returns the tools as a list of BaseToolProtocol objects.
        """
        return [
            GenericTool(
                name="query_graph",
                func=self.query_graph,
                description="Query the knowledge graph for a specific entity or keyword. Returns matching triplets.",
                args_schema=QueryGraphInput
            ),
            GenericTool(
                name="get_entity_neighbors",
                func=self.get_entity_neighbors,
                description="Get the neighbors of a specific entity within the graph. Supports depth and direction.",
                args_schema=EntityNeighborsInput
            ),
            GenericTool(
                name="search_triplets",
                func=self.search_triplets,
                description="Search for all triplets that have a specific predicate (relationship type).",
                args_schema=SearchTripletsInput
            ),
        ]
