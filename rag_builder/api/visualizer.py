import logging
from typing import List, Dict, Any
from rag_builder.storage.base import BaseGraphStorage, Triplet

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """
    Utility class to transform graph storage data into visualization-friendly formats.
    """

    @staticmethod
    def to_cytoscape_format(storage: BaseGraphStorage, limit: int = 100) -> Dict[str, Any]:
        """
        Converts triplets from the storage into a Cytoscape.js compatible format.
        
        Args:
            storage (BaseGraphStorage): The graph storage instance.
            limit (int): Maximum number of triplets to process to prevent OOM.
            
        Returns:
            Dict[str, Any]: A dictionary containing 'nodes' and 'edges' lists.
        """
        try:
            triplets = storage.get_all_triplets()
            # Apply limit to prevent OOM on large graphs
            triplets = triplets[:limit]
            
            nodes = {}
            edges = []
            
            for t in triplets:
                # Add subject node
                if t.subject not in nodes:
                    nodes[t.subject] = {
                        "data": {
                            "id": t.subject,
                            "label": t.subject,
                            **t.subject_properties
                        }
                    }
                
                # Add object node
                if t.object not in nodes:
                    nodes[t.object] = {
                        "data": {
                            "id": t.object,
                            "label": t.object,
                            **t.object_properties
                        }
                    }
                
                # Add edge
                edges.append({
                    "data": {
                        "id": f"{t.subject}-{t.predicate}-{t.object}",
                        "source": t.subject,
                        "target": t.object,
                        "label": t.predicate,
                        **t.predicate_properties
                    }
                })
                
            return {
                "nodes": list(nodes.values()),
                "edges": edges
            }
        except Exception as e:
            logger.error(f"Error exporting graph data to Cytoscape format: {e}")
            # We return an empty graph instead of raising to avoid breaking the UI, 
            # but in a production app, you might want to handle this differently.
            return {"nodes": [], "edges": []}
