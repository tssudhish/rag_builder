import logging
from typing import Any, Dict, List, Tuple, Set
from rag_builder.storage.base import BaseGraphStorage, GraphStats

logger = logging.getLogger(__name__)

class GraphInspector:
    """
    Provides high-level diagnostic and inspection queries for the knowledge graph.
    This class acts as a wrapper around BaseGraphStorage to provide analysis tools.
    """

    def __init__(self, storage: BaseGraphStorage) -> None:
        self.storage = storage

    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """
        Calculates the degree of a node (in-degree and out-degree).
        
        :param node_id: The identifier of the node to inspect.
        :return: A dictionary with 'in_degree' and 'out_degree'.
        """
        degree = self.storage.get_node_degree(node_id)
        return {
            "node_id": node_id,
            **degree,
            "total_degree": degree.get("in_degree", 0) + degree.get("out_degree", 0)
        }

    def get_relationship_distribution(self) -> Dict[str, int]:
        """
        Analyzes the distribution of relationship types (predicates) across the graph.
        
        :return: A dictionary mapping predicate names to their occurrence counts.
        """
        return self.storage.get_all_predicates()

    def get_connected_components_count(self) -> int:
        """
        Counts the number of connected components in the graph.
        
        :return: Number of connected components.
        """
        return self.storage.get_connected_components()

    def run_health_check(self) -> Dict[str, Any]:
        """
        Performs a comprehensive health check of the graph.
        
        :return: A dictionary containing various health metrics.
        """
        stats = self.storage.get_stats()
        orphans = self.storage.get_orphan_nodes()
        components = self.storage.get_connected_components()
        
        health = {
            "stats": stats.__dict__,
            "node_count": stats.node_count,
            "edge_count": stats.edge_count,
            "is_empty": stats.node_count == 0,
            "density": 0.0 if stats.node_count <= 1 else 
                       (2 * stats.edge_count) / (stats.node_count * (stats.node_count - 1)),
            "has_orphans": len(orphans) > 0,
            "orphan_count": len(orphans),
            "connected_components": components
        }
        
        return health

    def print_summary(self) -> None:
        """
        Prints a human-readable summary of the graph state to the console.
        Useful for CLI diagnostics.
        """
        stats = self.storage.get_stats()
        health = self.run_health_check()
        
        print("\n--- Knowledge Graph Inspection Summary ---")
        print(f"Nodes: {stats.node_count}")
        print(f"Edges: {stats.edge_count}")
        print(f"Unique Predicates: {stats.unique_predicates}")
        print(f"Backend: {stats.metadata.get('backend', 'Unknown')}")
        print(f"Is Empty: {health['is_empty']}")
        print(f"Density: {health['density']:.4f}")
        print(f"Connected Components: {health['connected_components']}")
        print(f"Orphan Nodes: {health['orphan_count']}")
        print("------------------------------------------\n")

