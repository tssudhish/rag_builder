from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Generator

@dataclass
class Triplet:
    """Represents a semantic triplet in the knowledge graph."""
    subject: str
    predicate: str
    object: str
    subject_properties: Dict[str, Any] = field(default_factory=dict)
    object_properties: Dict[str, Any] = field(default_factory=dict)
    predicate_properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphStats:
    """Statistics about the current state of the graph."""
    node_count: int
    edge_count: int
    unique_predicates: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseGraphStorage(ABC):
    """Abstract base class for graph database adapters."""

    @abstractmethod
    def insert_triplet(self, triplet: Triplet) -> None:
        """
        Inserts a triplet into the graph. 
        Should handle node and edge creation/updates.
        """
        pass

    @abstractmethod
    def query_neighbors(
        self, 
        node_id: str, 
        depth: int = 1, 
        direction: str = "both"
    ) -> Generator[Tuple[str, str, str], None, None]:
        """
        Retrieves neighbors of a given node up to a certain depth.
        Yields tuples of (source, predicate, target).
        
        :param node_id: The ID/name of the starting node.
        :param depth: How many hops to traverse.
        :param direction: 'in', 'out', or 'both'.
        """
        pass

    @abstractmethod
    def get_stats(self) -> GraphStats:
        """Returns basic statistics about the graph."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes the database connection if applicable."""
        pass
