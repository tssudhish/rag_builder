from typing import List, Dict, Optional, Tuple
import numpy as np
from rag_builder.storage.models import Entity

class VectorIndex:
    """
    Handles storage and retrieval of vector embeddings for entity nodes.
    In a production environment, this would interface with a vector DB like Milvus or Pinecone.
    Here, we implement an in-memory version using numpy for demonstration and testing.
    """
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index: Dict[str, np.ndarray] = {}  # entity_name -> embedding

    def add_embedding(self, entity: Entity, embedding: List[float]) -> None:
        """Adds or updates an embedding for a given entity."""
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {len(embedding)}")
        
        self.index[entity.name] = np.array(embedding, dtype=np.float32)

    def get_embedding(self, entity_name: str) -> Optional[np.ndarray]:
        """Retrieves the embedding for a specific entity."""
        return self.index.get(entity_name)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Performs a cosine similarity search across all indexed entities using vectorized operations.
        Returns a list of (entity_name, similarity_score) tuples.
        """
        if not self.index:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Vectorize search: stack all vectors into a matrix
        names = list(self.index.keys())
        matrix = np.array(list(self.index.values()), dtype=np.float32)
        
        # Compute norms for each vector in the matrix
        norms = np.linalg.norm(matrix, axis=1)
        
        # Guard against zero vectors in the index
        zero_mask = (norms == 0)
        # To avoid division by zero, we can replace 0 norms with 1 (similarity will be 0 anyway due to dot product)
        safe_norms = np.where(zero_mask, 1.0, norms)
        
        # Compute cosine similarity: (A . B) / (||A|| * ||B||)
        dot_products = np.dot(matrix, q_vec)
        similarities = dot_products / (q_norm * safe_norms)
        
        # Zero out similarities for zero vectors
        similarities[zero_mask] = 0.0

        # Combine names and scores, sort, and return top_k
        results = list(zip(names, similarities.astype(float)))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

    def remove_embedding(self, entity_name: str) -> None:
        """Removes an embedding from the index."""
        if entity_name in self.index:
            del self.index[entity_name]
