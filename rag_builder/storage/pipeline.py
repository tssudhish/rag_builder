from typing import List, Dict, Set, Tuple, Optional, Protocol, Any
from rag_builder.storage.models import Entity, Triplet
from rag_builder.storage.vector_index import VectorIndex

class GraphStore(Protocol):
    """Interface for the Graph Database storage."""
    def upsert_node(self, entity: Entity) -> None:
        ...
    def upsert_edge(self, triplet: Triplet) -> None:
        ...
    def get_all_nodes(self) -> List[Entity]:
        ...
    def get_all_edges(self) -> List[Triplet]:
        ...
    def contains_node(self, entity_name: str) -> bool:
        ...

class InMemoryGraphStore:
    """
    In-memory implementation of GraphStore for testing and development.
    In production, this would be replaced by a Neo4jGraphStore.
    """
    def __init__(self):
        self.nodes: Dict[str, Entity] = {}
        self.edges: Set[Tuple[str, str, str]] = set()  # (subject, predicate, object)
        self.edge_properties: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def upsert_node(self, entity: Entity) -> None:
        self.nodes[entity.name] = entity

    def upsert_edge(self, triplet: Triplet) -> None:
        edge_key = (triplet.subject.name, triplet.predicate, triplet.object.name)
        self.edges.add(edge_key)
        self.edge_properties[edge_key] = triplet.properties

    def get_all_nodes(self) -> List[Entity]:
        return list(self.nodes.values())

    def get_all_edges(self) -> List[Triplet]:
        triplets = []
        for edge_key in self.edges:
            subj_name, pred, obj_name = edge_key
            subj = self.nodes.get(subj_name)
            obj = self.nodes.get(obj_name)
            if subj and obj:
                triplets.append(Triplet(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    properties=self.edge_properties.get(edge_key, {})
                ))
        return triplets

    def contains_node(self, entity_name: str) -> bool:
        return entity_name in self.nodes

class IngestionPipeline:
    """
    Pipeline to ingest triplets into a Graph Store and a Vector Index.
    Handles deduplication and coordinate embedding generation (mocked here).
    """
    def __init__(self, graph_store: GraphStore, vector_index: VectorIndex):
        self.graph_store = graph_store
        self.vector_index = vector_index

    def _generate_embedding(self, entity: Entity) -> List[float]:
        """
        Generates a vector embedding for an entity.
        In production, this would call an embedding model (e.g., OpenAI, SentenceTransformers).
        For this implementation, we generate a deterministic pseudo-embedding.
        """
        import hashlib
        seed = entity.name.encode('utf-8')
        hash_val = int(hashlib.sha256(seed).hexdigest(), 16)
        
        # Generate a vector of length vector_index.dimension
        embedding = []
        for i in range(self.vector_index.dimension):
            # Simple deterministic generation based on hash
            val = ((hash_val >> (i % 64)) & 0xFF) / 255.0
            embedding.append(val)
        return embedding

    def ingest_batch(self, triplets: List[Triplet]) -> Dict[str, int]:
        """
        Processes a batch of triplets:
        1. Deduplicates entities and edges.
        2. Upserts nodes to the graph store.
        3. Generates and stores embeddings for new or updated nodes.
        4. Upserts edges to the graph store.
        
        Returns a summary of operations.
        """
        stats = {"nodes_added": 0, "edges_added": 0, "embeddings_created": 0}
        
        if not triplets:
            return stats

        processed_entities: Set[str] = set()
        processed_edges: Set[Tuple[str, str, str]] = set()

        for triplet in triplets:
            # Handle Subject
            subj = triplet.subject
            if subj.name not in processed_entities:
                # Only generate embedding if node doesn't already exist in the store
                # This prevents redundant compute across batches
                needs_embedding = not self.graph_store.contains_node(subj.name)
                
                self.graph_store.upsert_node(subj)
                
                if needs_embedding:
                    embedding = self._generate_embedding(subj)
                    self.vector_index.add_embedding(subj, embedding)
                    stats["embeddings_created"] += 1
                
                processed_entities.add(subj.name)
                if needs_embedding:
                    stats["nodes_added"] += 1

            # Handle Object
            obj = triplet.object
            if obj.name not in processed_entities:
                needs_embedding = not self.graph_store.contains_node(obj.name)
                
                self.graph_store.upsert_node(obj)
                
                if needs_embedding:
                    embedding = self._generate_embedding(obj)
                    self.vector_index.add_embedding(obj, embedding)
                    stats["embeddings_created"] += 1
                
                processed_entities.add(obj.name)
                if needs_embedding:
                    stats["nodes_added"] += 1

            # Handle Edge
            edge_key = (triplet.subject.name, triplet.predicate, triplet.object.name)
            if edge_key not in processed_edges:
                self.graph_store.upsert_edge(triplet)
                processed_edges.add(edge_key)
                stats["edges_added"] += 1

        return stats
