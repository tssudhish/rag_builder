import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_builder.storage.base import BaseGraphStorage

logger = logging.getLogger(__name__)

@dataclass
class SubGraphResult:
    """
    Contains the retrieved k-hop neighborhood around a target node.
    """
    target_node: str
    triplets: List[Tuple[str, str, str]] = field(default_factory=list)
    context_string: str = ""
    relevance_score: float = 0.0

class GraphRetriever:
    """
    Handles retrieval of structural context from the knowledge graph.
    """

    def __init__(self, storage: BaseGraphStorage):
        """
        Initialize the retriever with a graph storage backend.

        :param storage: An implementation of BaseGraphStorage.
        """
        self.storage = storage

    def retrieve_k_hop_neighborhood(
        self, 
        target_node: str, 
        k: int = 1, 
        direction: str = "both",
        scoring_fn=None
    ) -> SubGraphResult:
        """
        Fetches the target node and its k-hop neighbors.

        :param target_node: The starting node for retrieval.
        :param k: Number of hops to traverse.
        :param direction: 'in', 'out', or 'both'.
        :param scoring_fn: Optional function to score the relevance of the subgraph.
        :return: A SubGraphResult containing the neighborhood and a formatted context.
        """
        try:
            # BaseGraphStorage.query_neighbors returns a generator of (subject, predicate, object)
            triplets = list(self.storage.query_neighbors(
                node_id=target_node, 
                depth=k, 
                direction=direction
            ))
            
            # Calculate relevance score if scoring_fn is provided, otherwise default to 1.0 if results exist
            score = 1.0 if triplets else 0.0
            if scoring_fn:
                score = scoring_fn(target_node, triplets)

            result = SubGraphResult(
                target_node=target_node,
                triplets=triplets,
                relevance_score=score
            )
            
            # Format as structured prompt context
            result.context_string = self._format_as_context(result)
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving k-hop neighborhood for {target_node}: {e}")
            return SubGraphResult(target_node=target_node, relevance_score=0.0)

    def _format_as_context(self, result: SubGraphResult) -> str:
        """
        Converts a list of triplets into a structured string for LLM context.
        """
        if not result.triplets:
            return f"No structural context found for entity: {result.target_node}"

        lines = [f"Knowledge Graph Context for '{result.target_node}':"]
        for subject, predicate, obj in result.triplets:
            lines.append(f"- {subject} --({predicate})--> {obj}")
        
        return "\n".join(lines)
