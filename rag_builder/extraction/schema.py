from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    OBJECT = "OBJECT"
    DATE = "DATE"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class Entity:
    """Represents a normalized entity in the knowledge graph."""
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure name is not empty
        if not self.name or not self.name.strip():
            raise ValueError("Entity name cannot be empty.")

@dataclass(frozen=True)
class Relation:
    """Represents a normalized relation between two entities."""
    predicate: str
    relation_type: str = "RELATION"  # Categorical type of relation (e.g., 'causal', 'membership')
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.predicate or not self.predicate.strip():
            raise ValueError("Relation predicate cannot be empty.")

@dataclass(frozen=True)
class Triplet:
    """A validated knowledge triplet with confidence scores."""
    subject: Entity
    predicate: Relation
    object: Entity
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")

# Alias for explicit clarity when distinguishing from raw extracted triplets
ValidatedTriplet = Triplet
