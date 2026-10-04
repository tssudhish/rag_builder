from typing import List, Dict, Any, Tuple, Generator
from rag_builder.storage.base import BaseGraphStorage, Triplet, GraphStats
from collections import defaultdict

class MemoryGraphStorage(BaseGraphStorage):
    """
    In-memory implementation of Graph Storage for development and testing.
    """
    def __init__(self):
        self._triplets: List[Triplet] = []
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def insert_triplet(self, triplet: Triplet) -> None:
        self._triplets.append(triplet)
        
        # Update node properties (merge)
        if triplet.subject not in self._nodes:
            self._nodes[triplet.subject] = {}
        self._nodes[triplet.subject].update(triplet.subject_properties)
        
        if triplet.object not in self._nodes:
            self._nodes[triplet.object] = {}
        self._nodes[triplet.object].update(triplet.object_properties)

    def query_neighbors(
        self, 
        node_id: str, 
        depth: int = 1, 
        direction: str = "both"
    ) -> Generator[Tuple[str, str, str], None, None]:
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_node, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
                
            for t in self._triplets:
                if direction in ("out", "both") and t.subject == current_node:
                    yield (t.subject, t.predicate, t.object)
                    if t.object not in visited:
                        visited.add(t.object)
                        queue.append((t.object, current_depth + 1))
                
                if direction in ("in", "both") and t.object == current_node:
                    yield (t.subject, t.predicate, t.object)
                    if t.subject not in visited:
                        visited.add(t.subject)
                        queue.append((t.subject, current_depth + 1))

    def get_stats(self) -> GraphStats:
        predicates = {t.predicate for t in self._triplets}
        return GraphStats(
            node_count=len(self._nodes),
            edge_count=len(self._triplets),
            unique_predicates=len(predicates)
        )

    def get_all_predicates(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for t in self._triplets:
            counts[t.predicate] += 1
        return dict(counts)

    def get_connected_components(self) -> int:
        visited = set()
        components = 0
        
        all_nodes = set(self._nodes.keys())
        for node in all_nodes:
            if node not in visited:
                components += 1
                # BFS to find all connected nodes
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        # Find all neighbors
                        for t in self._triplets:
                            if t.subject == curr:
                                stack.append(t.object)
                            elif t.object == curr:
                                stack.append(t.subject)
        return components

    def get_orphan_nodes(self) -> List[str]:
        connected_nodes = set()
        for t in self._triplets:
            connected_nodes.add(t.subject)
            connected_nodes.add(t.object)
        
        return [node for node in self._nodes if node not in connected_nodes]

    def get_node_degree(self, node_id: str) -> Dict[str, int]:
        in_degree = 0
        out_degree = 0
        for t in self._triplets:
            if t.subject == node_id:
                out_degree += 1
            if t.object == node_id:
                in_degree += 1
        return {"in_degree": in_degree, "out_degree": out_degree}

    def get_all_triplets(self) -> List[Triplet]:
        return self._triplets

    def close(self) -> None:
        pass
