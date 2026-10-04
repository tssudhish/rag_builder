import logging
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Generator
from rag_builder.storage.base import BaseGraphStorage, Triplet, GraphStats

logger = logging.getLogger(__name__)

class MemoryGraphStorage(BaseGraphStorage):
    """
    In-memory graph storage implementation using NetworkX.
    Used for lightweight local testing and small-scale RAG.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def insert_triplet(self, triplet: Triplet) -> None:
        """
        Inserts a triplet into the in-memory graph.
        Nodes are identified by their subject/object strings.
        """
        try:
            # Add nodes with properties
            self._graph.add_node(triplet.subject, **triplet.subject_properties)
            self._graph.add_node(triplet.object, **triplet.object_properties)
            
            # Add edge with predicate as the key/attribute
            # We use the predicate as an attribute for easier querying
            self._graph.add_edge(
                triplet.subject, 
                triplet.object, 
                predicate=triplet.predicate, 
                **triplet.predicate_properties
            )
        except Exception as e:
            logger.error(f"Failed to insert triplet {triplet}: {e}")
            raise

    def query_neighbors(
        self, 
        node_id: str, 
        depth: int = 1, 
        direction: str = "both"
    ) -> Generator[Tuple[str, str, str], None, None]:
        """
        Retrieves neighbors of a given node up to a certain depth.
        """
        if node_id not in self._graph:
            logger.warning(f"Node {node_id} not found in memory graph.")
            return

        # Use a set to avoid duplicate triplets when traversing
        yielded_triplets = set()

        if direction == "both":
            undirected = self._graph.to_undirected()
            distances = nx.single_source_shortest_path_length(undirected, node_id, cutoff=depth)
            for node in distances:
                for neighbor in undirected.neighbors(node):
                    if neighbor in distances:
                        # Original edges: node -> neighbor
                        edges_out = self._graph.get_edge_data(node, neighbor)
                        if edges_out:
                            for key, data in edges_out.items():
                                triplet = (node, data.get("predicate", "UNKNOWN"), neighbor)
                                if triplet not in yielded_triplets:
                                    yielded_triplets.add(triplet)
                                    yield triplet
                        # Original edges: neighbor -> node
                        edges_in = self._graph.get_edge_data(neighbor, node)
                        if edges_in and node != neighbor:
                            for key, data in edges_in.items():
                                triplet = (neighbor, data.get("predicate", "UNKNOWN"), node)
                                if triplet not in yielded_triplets:
                                    yielded_triplets.add(triplet)
                                    yield triplet
            return

        graph_to_query = self._graph if direction == "out" else self._graph.reverse()
        distances = nx.single_source_shortest_path_length(graph_to_query, node_id, cutoff=depth)
        
        for node in distances:
            for neighbor in graph_to_query.neighbors(node):
                if neighbor in distances:
                    if direction == "out":
                        edges = self._graph.get_edge_data(node, neighbor)
                        if edges:
                            for key, data in edges.items():
                                triplet = (node, data.get("predicate", "UNKNOWN"), neighbor)
                                if triplet not in yielded_triplets:
                                    yielded_triplets.add(triplet)
                                    yield triplet
                    else: # direction == "in"
                        edges = self._graph.get_edge_data(neighbor, node)
                        if edges:
                            for key, data in edges.items():
                                triplet = (neighbor, data.get("predicate", "UNKNOWN"), node)
                                if triplet not in yielded_triplets:
                                    yielded_triplets.add(triplet)
                                    yield triplet


    def get_stats(self) -> GraphStats:
        """Returns statistics about the memory graph."""
        return GraphStats(
            node_count=self._graph.number_of_nodes(),
            edge_count=self._graph.number_of_edges(),
            unique_predicates=len(set(
                data.get("predicate") 
                for _, _, data in self._graph.edges(data=True) 
                if "predicate" in data
            )),
            metadata={"backend": "NetworkX"}
        )

    def close(self) -> None:
        """No-op for in-memory storage."""
        pass
