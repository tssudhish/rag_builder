import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Generator
from neo4j import GraphDatabase, Driver
from rag_builder.storage.base import BaseGraphStorage, Triplet, GraphStats

logger = logging.getLogger(__name__)

class Neo4jGraphStorage(BaseGraphStorage):
    """
    Neo4j graph storage adapter supporting Cypher queries and connection pooling.
    """

    def __init__(
        self, 
        uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password"
    ) -> None:
        try:
            self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {uri}: {e}")
            raise

    def _sanitize_predicate(self, predicate: str) -> str:
        """
        Sanitizes predicates to prevent Cypher injection.
        Only allows alphanumeric characters and underscores.
        """
        return re.sub(r'[^a-zA-Z0-9_]', '_', predicate)

    def insert_triplet(self, triplet: Triplet) -> None:
        """
        Inserts a triplet into Neo4j. 
        Uses MERGE to ensure uniqueness of nodes and edges.
        """
        # Sanitize predicate for use as a Relationship type
        safe_predicate = self._sanitize_predicate(triplet.predicate)
        
        # Construct Cypher query
        # Use parameters for node properties to avoid injection
        # Relationship type cannot be parameterized in Cypher, hence the sanitization
        query = (
            f"MERGE (s:Entity {{name: $subject_name}}) "
            f"SET s += $subject_props "
            f"MERGE (o:Entity {{name: $object_name}}) "
            f"SET o += $object_props "
            f"MERGE (s)-[r:{safe_predicate}]->(o) "
            f"SET r += $rel_props"
        )
        
        parameters = {
            "subject_name": triplet.subject,
            "subject_props": triplet.subject_properties,
            "object_name": triplet.object,
            "object_props": triplet.object_properties,
            "rel_props": triplet.predicate_properties
        }

        try:
            with self._driver.session() as session:
                session.run(query, parameters)
        except Exception as e:
            logger.error(f"Failed to insert triplet into Neo4j: {e}")
            raise

    def query_neighbors(
        self, 
        node_id: str, 
        depth: int = 1, 
        direction: str = "both"
    ) -> Generator[Tuple[str, str, str], None, None]:
        """
        Retrieves neighbors of a given node up to a certain depth.
        Handles paths for depth > 1 by iterating through segments.
        """
        # Define direction in Cypher
        # both: (n)-[r]-(m)
        # out: (n)-[r]->(m)
        # in: (n)<-[r]-(m)
        dir_pattern = ""
        if direction == "out":
            dir_pattern = "-->"
        elif direction == "in":
            dir_pattern = "<--"
        else:
            dir_pattern = "-"

        query = (
            f"MATCH p = (n:Entity {{name: $node_id}}){dir_pattern}*[1..{depth}]-(m:Entity) "
            f"RETURN p"
        )

        try:
            with self._driver.session() as session:
                result = session.run(query, node_id=node_id)
                for record in result:
                    path = record["p"]
                    # A path consists of segments (nodes and relationships)
                    # We iterate through segments to yield each individual triplet
                    for segment in path.relationships:
                        start_node = segment.start_node["name"]
                        end_node = segment.end_node["name"]
                        rel_type = segment.type
                        yield (start_node, rel_type, end_node)
        except Exception as e:
            logger.error(f"Error querying neighbors in Neo4j: {e}")
            return

    def get_stats(self) -> GraphStats:
        """Returns statistics about the Neo4j graph."""
        query = (
            "MATCH (n:Entity) WITH count(n) as node_count "
            "MATCH ()-[r]->() WITH node_count, count(r) as edge_count "
            "MATCH ()-[r]->() WITH node_count, edge_count, count(DISTINCT type(r)) as pred_count "
            "RETURN node_count, edge_count, pred_count"
        )
        
        try:
            with self._driver.session() as session:
                result = session.run(query).single()
                if result:
                    return GraphStats(
                        node_count=result["node_count"],
                        edge_count=result["edge_count"],
                        unique_predicates=result["pred_count"],
                        metadata={"backend": "Neo4j"}
                    )
                else:
                    raise ValueError("No result returned from stats query")
        except Exception as e:
            logger.error(f"Error getting stats from Neo4j: {e}")
            # Return empty stats or handle as per reviewer request
            return GraphStats(0, 0, 0, {"error": str(e)})

    def close(self) -> None:
        """Closes the Neo4j driver."""
        if self._driver:
            self._driver.close()
