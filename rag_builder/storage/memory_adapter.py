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

    def get_all_predicates(self) -> Dict[str, int]:
        """Returns a distribution of all predicate types in the memory graph."""
        distribution = {}
        for _, _, data in self._graph.edges(data=True):
            pred = data.get("predicate", "UNKNOWN")
            distribution[pred] = distribution.get(pred, 0) + 1
        return distribution

    def get_connected_components(self) -> int:
        """Returns the number of connected components in the memory graph."""
        undirected = self._graph.to_undirected()
        return nx.number_connected_components(undirected)

    def get_orphan_nodes(self) -> List[str]:
        """Returns a list of nodes with no edges in the memory graph."""
        return [node for node, degree in self._graph.degree() if degree == 0]

    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        """Returns the in-degree and out-degree of a node in the memory graph."""
        if node_id not in self._graph:
            return {"in_degree": 0, "out_degree": 0}
        return {
            "in_degree": self._graph.in_degree(node_id),
            "out_degree": self._graph.out_degree(node_id)
        }

    def get_all_triplets(self) -> List[Triplet]:
        """Returns all triplets currently stored in the memory graph."""
        triplets = []
        for u, v, data in self._graph.edges(data=True):
            # Find properties for nodes
            u_props = self._graph.nodes[u] if u in self._graph.nodes else {}
            v_props = self._graph.nodes[v] if v in self._graph.nodes else {}
            
            triplets.append(Triplet(
                subject=u,
                predicate=data.get("predicate", "UNKNOWN"),
                object=v,
                subject_properties=u_props,
                object_properties=v_props,
                predicate_properties={k: v for k, v in data.items() if k != "predicate"}
            ))
        return triplets

    def export_json(self, file_path: str) -> None:
        """
        Exports the current graph to a JSON file.
        """
        import json
        triplets = self.get_all_triplets()
        # Convert Triplet dataclasses to dicts for JSON serialization
        data = [
            {
                "subject": t.subject,
                "predicate": t.predicate,
                "object": t.object,
                "subject_properties": t.subject_properties,
                "object_properties": t.object_properties,
                "predicate_properties": t.predicate_properties,
            }
            for t in triplets
        ]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def import_json(self, file_path: str) -> None:
        """
        Imports triplets from a JSON file and populates the graph.
        """
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            self.insert_triplet(Triplet(
                subject=item["subject"],
                predicate=item["predicate"],
                object=item["object"],
                subject_properties=item.get("subject_properties", {}),
                object_properties=item.get("object_properties", {}),
                predicate_properties=item.get("predicate_properties", {}),
            ))



    def close(self) -> None:
        """No-op for in-memory storage."""
        pass
