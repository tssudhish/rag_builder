import os
import requests
import json
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class QueryGenerator:
    """
    Converts natural language questions into graph retrieval queries (Cypher).
    Uses Ollama for LLM-based translation and implements strict safety checks.
    """

    ALLOWED_START_CLAUSES = ("MATCH", "OPTIONAL MATCH", "WITH", "RETURN", "UNWIND")
    FORBIDDEN_CYPHER_TOKENS = (";", "//", "DELETE", "DETACH DELETE", "CREATE", "MERGE", "SET", "REMOVE")

    def __init__(
        self, 
        ollama_url: Optional[str] = None, 
        model: Optional[str] = None
    ):
        """
        Initialize the QueryGenerator.
        
        Args:
            ollama_url: URL of the Ollama server. Fallback to OLLAMA_HOST env var or default.
            model: The model to use for translation. Fallback to OLLAMA_MODEL env var or default.
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2")
        
        # Ensure URL ends with /api/generate
        if not self.ollama_url.endswith("/api/generate"):
            base_url = self.ollama_url.rstrip("/")
            self.ollama_url = f"{base_url}/api/generate"

    def generate_query(self, question: str, schema_info: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Translates a natural language question into a Cypher query.
        
        Args:
            question: The user's question.
            schema_info: Optional description of the graph schema to guide the LLM.
            
        Returns:
            A tuple of (cypher_query, parameters).
            
        Raises:
            ValueError: If the generated query is invalid or unsafe.
        """
        prompt = self._build_prompt(question, schema_info)
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            content = result.get("response", "")
            
            # Parse JSON response from LLM
            data = json.loads(content)
            cypher = data.get("cypher", "")
            params = data.get("parameters", {})
            
            # Safety Validations
            self._validate_generated_cypher(cypher)
            self._validate_parameters(params)
            
            return cypher, params
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Error during query generation: {str(e)}")
            raise RuntimeError(f"Failed to communicate with Ollama or parse response: {e}")

    def _build_prompt(self, question: str, schema_info: Optional[str]) -> str:
        """Constructs the prompt for the LLM."""
        schema_context = schema_info if schema_info else "General knowledge graph schema."
        
        return (
            f"You are a Cypher query expert. Convert the following natural language question into a read-only Cypher query.\n"
            f"Schema context: {schema_context}\n"
            f"Question: {question}\n\n"
            f"Respond ONLY in JSON format with two keys: 'cypher' and 'parameters'.\n"
            f"The 'cypher' value must be a valid read-only query using parameters (e.g., $name).\n"
            f"The 'parameters' value must be a dictionary of parameter values.\n"
            f"Example: {{\"cypher\": \"MATCH (n:Person {{name: $name}}) RETURN n\", \"parameters\": {{\"name\": \"Alice\"}}}}\n"
            f"Strictly avoid any write operations (CREATE, DELETE, SET, MERGE)."
        )

    def _validate_generated_cypher(self, cypher: str) -> None:
        """
        Strictly validates that the generated Cypher query is a read-only operation.
        Fails closed if any unsafe patterns are detected.
        """
        if not cypher:
            raise ValueError("Generated Cypher query is empty.")

        trimmed_cypher = cypher.strip().upper()
        
        # 1. Strict Validation: Must start with allowed read clauses
        if not trimmed_cypher.startswith(self.ALLOWED_START_CLAUSES):
            raise ValueError(
                f"Unsafe Cypher query detected. Query must start with one of {self.ALLOWED_START_CLAUSES}. "
                f"Received: {trimmed_cypher[:20]}..."
            )
            
        # 2. Block destructive tokens/commands
        for token in self.FORBIDDEN_CYPHER_TOKENS:
            if token in trimmed_cypher:
                raise ValueError(f"Unsafe Cypher token detected: {token}")

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Ensures that parameter values are safe primitives and do not contain injection patterns.
        """
        if not isinstance(params, dict):
            raise ValueError("Parameters must be a dictionary.")
            
        safe_types = (str, int, float, bool, list, type(None))
        
        for key, value in params.items():
            if not isinstance(value, safe_types):
                raise ValueError(f"Parameter {key} has an unsafe type: {type(value)}")
            
            if isinstance(value, str):
                # Check for Cypher statement separators or comment markers in string values
                if any(marker in value for marker in (";", "//")):
                    raise ValueError(f"Parameter {key} contains unsafe characters (';' or '//')")
            
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, safe_types):
                        raise ValueError(f"List item in parameter {key} has an unsafe type: {type(item)}")
                    if isinstance(item, str) and any(marker in item for marker in (";", "//")):
                        raise ValueError(f"List item in parameter {key} contains unsafe characters")
