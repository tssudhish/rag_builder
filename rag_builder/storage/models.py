from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class Entity:
    """Represents a node in the knowledge graph."""
    name: str
    label: str  # e.g., "Person", "Organization", "Concept"
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Triplet:
    """Represents a directed edge between two entities."""
    subject: Entity
    predicate: str  # e.g., "works_for", "is_a"
    object: Entity
    properties: Dict[str, Any] = field(default_factory=dict)
