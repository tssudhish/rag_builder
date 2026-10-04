import pytest
from rag_builder.extraction.schema import Entity, Relation, Triplet, EntityType
from rag_builder.extraction.normalizer import (
    EntityNormalizer,
    RelationNormalizer,
    SchemaNormalizer,
)

@pytest.fixture
def normalizer_setup():
    entity_aliases = {
        "usa": "United States",
        "us": "United States",
        "elon": "Elon Musk",
        "tesla motors": "Tesla",
        "spacex": "SpaceX",
        "tim cook": "Tim Cook",
    }
    relation_synonyms = {
        "ceo of": "LEADS",
        "founded": "FOUNDED",
    }
    entity_norm = EntityNormalizer(entity_aliases)
    relation_norm = RelationNormalizer(relation_synonyms)
    schema_norm = SchemaNormalizer(
        entity_aliases=entity_aliases,
        relation_synonyms=relation_synonyms,
    )
    return {
        "entity_norm": entity_norm,
        "relation_norm": relation_norm,
        "schema_norm": schema_norm,
    }

def test_entity_alias_mapping(normalizer_setup):
    """Verify alias map resolution and whitespace trimming."""
    norm = normalizer_setup["entity_norm"]
    assert norm.normalize_name("  usa  ") == "United States"
    assert norm.normalize_name("US") == "United States"
    assert norm.normalize_name("elon") == "Elon Musk"

def test_entity_casing_preservation(normalizer_setup):
    """Verify that acronyms and camelCase are preserved by default."""
    norm = normalizer_setup["entity_norm"]
    assert norm.normalize_name("NASA") == "NASA"
    assert norm.normalize_name("iPhone") == "iPhone"
    assert norm.normalize_name("OpenAI") == "OpenAI"

def test_entity_auto_titlecase():
    """Verify that auto_titlecase=True applies title casing."""
    norm = EntityNormalizer(auto_titlecase=True)
    assert norm.normalize_name("albert einstein") == "Albert Einstein"

def test_cross_chunk_canonicalization(normalizer_setup):
    """Verify that entities are canonicalized based on first seen occurrence."""
    norm = normalizer_setup["entity_norm"]
    norm.register_entity("OpenAI")
    
    assert norm.normalize_name("openai") == "OpenAI"
    assert norm.normalize_name("OPENAI") == "OpenAI"
    
    norm.register_entity("Microsoft")
    assert norm.normalize_name("microsoft") == "Microsoft"

def test_discover_entities(normalizer_setup):
    """Verify discover_entities populates the registry correctly."""
    norm = normalizer_setup["entity_norm"]
    entities = [
        Entity(name="DeepMind", entity_type=EntityType.ORGANIZATION),
        Entity(name="Google", entity_type=EntityType.ORGANIZATION),
    ]
    norm.discover_entities(entities)
    assert norm.normalize_name("deepmind") == "DeepMind"
    assert norm.normalize_name("google") == "Google"

def test_entity_object_normalization(normalizer_setup):
    """Verify normalization preserves entity type and metadata."""
    norm = normalizer_setup["entity_norm"]
    entity = Entity(name="tesla motors", entity_type=EntityType.ORGANIZATION, metadata={"source": "doc1"})
    norm_entity = norm.normalize_entity(entity)
    assert norm_entity.name == "Tesla"
    assert norm_entity.entity_type == EntityType.ORGANIZATION
    assert norm_entity.metadata["source"] == "doc1"

def test_relation_synonym_mapping(normalizer_setup):
    """Verify relation synonym map resolution, including defaults."""
    norm = normalizer_setup["relation_norm"]
    assert norm.normalize_predicate("works at") == "EMPLOYED_BY"
    assert norm.normalize_predicate("  is employed by  ") == "EMPLOYED_BY"
    assert norm.normalize_predicate("located in") == "LOCATED_IN"
    assert norm.normalize_predicate("is part of") == "PART_OF"
    assert norm.normalize_predicate("ceo of") == "LEADS"

def test_relation_snake_case_fallback(normalizer_setup):
    """Verify unmapped relations are converted to UPPER_SNAKE_CASE."""
    norm = normalizer_setup["relation_norm"]
    assert norm.normalize_predicate("invested in") == "INVESTED_IN"
    assert norm.normalize_predicate("headquartered-in") == "HEADQUARTERED_IN"

def test_schema_normalizer_single_triplet(normalizer_setup):
    """Verify end-to-end normalization of a full Triplet."""
    norm = normalizer_setup["schema_norm"]
    subj = Entity(name="elon", entity_type=EntityType.PERSON)
    rel = Relation(predicate="ceo of")
    obj = Entity(name="tesla motors", entity_type=EntityType.ORGANIZATION)
    triplet = Triplet(subject=subj, predicate=rel, object=obj, confidence=0.95)

    norm_triplet = norm.normalize_triplet(triplet)
    assert norm_triplet.subject.name == "Elon Musk"
    assert norm_triplet.predicate.predicate == "LEADS"
    assert norm_triplet.object.name == "Tesla"
    assert norm_triplet.confidence == 0.95

def test_schema_normalizer_batch(normalizer_setup):
    """Verify batch normalization across multiple triplets."""
    norm = normalizer_setup["schema_norm"]
    t1 = Triplet(
        subject=Entity(name="elon"),
        predicate=Relation(predicate="founded"),
        object=Entity(name="spacex"),
    )
    t2 = Triplet(
        subject=Entity(name="tim cook"),
        predicate=Relation(predicate="works at"),
        object=Entity(name="apple"),
    )
    batch = norm.normalize_batch([t1, t2])
    assert len(batch) == 2
    assert batch[0].subject.name == "Elon Musk"
    assert batch[0].object.name == "SpaceX"
    assert batch[1].predicate.predicate == "EMPLOYED_BY"
    assert batch[1].subject.name == "Tim Cook"

def test_relation_linguistic_variations(normalizer_setup):
    """Verify linguistic variations map to canonical relations."""
    norm = normalizer_setup["relation_norm"]
    assert norm.normalize_predicate("is a part of") == "PART_OF"
    assert norm.normalize_predicate("is a member of") == "MEMBER_OF"
    assert norm.normalize_predicate("member of") == "MEMBER_OF"

def test_two_pass_cross_chunk_normalization():
    """Verify prepare_registry warms up cross-chunk entity names for subsequent normalization."""
    norm = SchemaNormalizer()
    chunk1_triplets = [
        Triplet(
            subject=Entity(name="OpenAI"),
            predicate=Relation(predicate="developed"),
            object=Entity(name="ChatGPT"),
        )
    ]
    chunk2_triplets = [
        Triplet(
            subject=Entity(name="openai"),
            predicate=Relation(predicate="released"),
            object=Entity(name="chatgpt"),
        )
    ]
    # Pass 1: prepare registry from all chunks
    norm.prepare_registry(chunk1_triplets + chunk2_triplets)

    # Pass 2: normalize chunk 2
    normalized_chunk2 = norm.normalize_batch(chunk2_triplets)
    assert normalized_chunk2[0].subject.name == "OpenAI"
    assert normalized_chunk2[0].object.name == "ChatGPT"

def test_schema_validation_constraints():
    """Verify validation errors on invalid entities, relations, or confidences."""
    with pytest.raises(ValueError):
        Entity(name="")
    with pytest.raises(ValueError):
        Entity(name="   ")
    with pytest.raises(ValueError):
        Relation(predicate="")
    with pytest.raises(ValueError):
        Triplet(
            subject=Entity(name="Valid"),
            predicate=Relation(predicate="VALID"),
            object=Entity(name="Valid"),
            confidence=1.5,
        )
