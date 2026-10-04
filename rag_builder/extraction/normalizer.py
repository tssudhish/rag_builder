import re
from typing import Dict, List, Optional, Set
from rag_builder.extraction.schema import Entity, Relation, Triplet, EntityType

class EntityNormalizer:
    """
    Handles normalization of entity names to prevent duplication.
    Implements casing standardization and canonical alias mapping.
    Supports cross-chunk canonicalization by tracking seen entities in a registry.
    """
    def __init__(self, alias_map: Optional[Dict[str, str]] = None, auto_titlecase: bool = False):
        """
        Initialize the EntityNormalizer.
        
        Args:
            alias_map: A dictionary mapping aliases to canonical names.
                       Example: {"USA": "United States", "US": "United States"}
            auto_titlecase: If True, applies .title() to names not found in alias map or registry.
                            Defaults to False to preserve acronyms and camelCase.
        """
        self.alias_map = {k.strip().lower(): v.strip() for k, v in (alias_map or {}).items()}
        self.auto_titlecase = auto_titlecase
        # registry maps lowercased names to their canonical (original discovered) casing
        self.registry: Dict[str, str] = {}

    def register_entity(self, name: str, canonical_name: Optional[str] = None) -> str:
        """
        Registers an entity name in the registry.
        If canonical_name is provided, it maps the name to that canonical version.
        Otherwise, it preserves the first or best capitalized representation.
        """
        clean_name = name.strip()
        lowered = clean_name.lower()
        canonical = canonical_name.strip() if canonical_name else clean_name
        
        if canonical_name or lowered not in self.registry:
            self.registry[lowered] = canonical
        elif self.registry[lowered].islower() and not canonical.islower():
            # Upgrade from all-lowercase to capitalized/camelCase canonical form
            self.registry[lowered] = canonical
            
        return self.registry[lowered]

    def discover_entities(self, entities: List[Entity]):
        """
        Processes a list of entities to populate the registry before normalization.
        This helps in establishing canonical casing based on first occurrence.
        """
        for entity in entities:
            self.register_entity(entity.name)

    def normalize_name(self, name: str) -> str:
        """
        Normalizes a single entity name.
        
        1. Strips whitespace.
        2. Checks against alias map (case-insensitive).
        3. Checks against registry (case-insensitive).
        4. Applies title casing only if auto_titlecase=True and no match found.
        """
        clean_name = name.strip()
        lowered = clean_name.lower()
        
        # 1. Check for canonical alias
        if lowered in self.alias_map:
            return self.alias_map[lowered]
            
        # 2. Check registry for cross-chunk canonicalization
        if lowered in self.registry:
            return self.registry[lowered]
            
        # 3. Handle fallback
        normalized_name = clean_name
        if self.auto_titlecase:
            normalized_name = clean_name.title()
            
        return normalized_name

    def normalize_entity(self, entity: Entity) -> Entity:
        """Returns a new Entity instance with a normalized name."""
        normalized_name = self.normalize_name(entity.name)
        if normalized_name == entity.name:
            return entity
        
        return Entity(
            name=normalized_name,
            entity_type=entity.entity_type,
            metadata=entity.metadata
        )

class RelationNormalizer:
    """
    Handles normalization of relation predicates to standardize synonyms.
    """
    def __init__(self, synonym_map: Optional[Dict[str, str]] = None):
        """
        Initialize the RelationNormalizer.
        
        Args:
            synonym_map: Mapping of synonymous predicates to a canonical one.
                          Example: {"works at": "EMPLOYED_BY", "is employed by": "EMPLOYED_BY"}
        """
        # Standard default synonyms and common linguistic variations
        defaults = {
            "is part of": "PART_OF",
            "is a part of": "PART_OF",
            "part of": "PART_OF",
            "located in": "LOCATED_IN",
            "is located in": "LOCATED_IN",
            "works at": "EMPLOYED_BY",
            "is employed by": "EMPLOYED_BY",
            "employed by": "EMPLOYED_BY",
            "is member of": "MEMBER_OF",
            "is a member of": "MEMBER_OF",
            "member of": "MEMBER_OF",
        }
        
        if synonym_map:
            defaults.update({k.strip().lower(): v.strip().upper() for k, v in synonym_map.items()})
            
        self.synonym_map = defaults

    def normalize_predicate(self, predicate: str) -> str:
        """
        Normalizes a predicate string.
        
        1. Lowercases and strips.
        2. Maps to canonical relation via synonym map.
        3. If no mapping, converts to UPPER_SNAKE_CASE.
        """
        clean_pred = predicate.strip().lower()
        
        # Check synonym map
        normalized = self.synonym_map.get(clean_pred)
        if normalized:
            return normalized
            
        # Fallback: Convert "works at" or "is CEO of" -> "WORKS_AT", "IS_CEO_OF"
        return re.sub(r'[\s\-]+', '_', clean_pred).upper()

    def normalize_relation(self, relation: Relation) -> Relation:
        """Returns a new Relation instance with a normalized predicate."""
        normalized_pred = self.normalize_predicate(relation.predicate)
        if normalized_pred == relation.predicate:
            return relation
            
        return Relation(
            predicate=normalized_pred,
            relation_type=relation.relation_type,
            metadata=relation.metadata
        )

class SchemaNormalizer:
    """
    High-level coordinator that applies both Entity and Relation normalization
    to a set of Triplets.
    Supports a two-pass architecture:
      Pass 1: prepare_registry(triplets) across all chunks to discover and establish canonical entities.
      Pass 2: normalize_batch(triplets) to canonicalize and format all triplets.
    """
    def __init__(
        self, 
        entity_aliases: Optional[Dict[str, str]] = None, 
        relation_synonyms: Optional[Dict[str, str]] = None,
        auto_titlecase: bool = False
    ):
        self.entity_normalizer = EntityNormalizer(entity_aliases, auto_titlecase=auto_titlecase)
        self.relation_normalizer = RelationNormalizer(relation_synonyms)

    def prepare_registry(self, all_triplets: List[Triplet]) -> None:
        """
        Pass 1: Warms up the entity registry across all chunks.
        Extracts all subjects and objects from the entire corpus to establish
        canonical casings and references prior to final normalization.
        """
        entities = []
        for t in all_triplets:
            entities.append(t.subject)
            entities.append(t.object)
        self.entity_normalizer.discover_entities(entities)

    def normalize_triplet(self, triplet: Triplet) -> Triplet:
        """Normalizes all components of a single triplet."""
        norm_subject = self.entity_normalizer.normalize_entity(triplet.subject)
        norm_object = self.entity_normalizer.normalize_entity(triplet.object)
        norm_predicate = self.relation_normalizer.normalize_relation(triplet.predicate)
        
        return Triplet(
            subject=norm_subject,
            predicate=norm_predicate,
            object=norm_object,
            confidence=triplet.confidence,
            source_chunk_id=triplet.source_chunk_id,
            metadata=triplet.metadata
        )

    def normalize_batch(self, triplets: List[Triplet]) -> List[Triplet]:
        """Normalizes a list of triplets."""
        return [self.normalize_triplet(t) for t in triplets]

