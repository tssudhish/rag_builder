import unittest
from rag_builder.extraction.schema import Entity, Relation, Triplet, EntityType
from rag_builder.extraction.normalizer import (
    EntityNormalizer,
    RelationNormalizer,
    SchemaNormalizer,
)

class TestSchemaAndNormalization(unittest.TestCase):
    def setUp(self):
        self.entity_aliases = {
            "usa": "United States",
            "us": "United States",
            "elon": "Elon Musk",
            "tesla motors": "Tesla",
            "spacex": "SpaceX",
            "tim cook": "Tim Cook",
        }
        self.relation_synonyms = {
            "ceo of": "LEADS",
            "founded": "FOUNDED",
        }
        self.entity_norm = EntityNormalizer(self.entity_aliases)
        self.relation_norm = RelationNormalizer(self.relation_synonyms)
        self.schema_norm = SchemaNormalizer(
            entity_aliases=self.entity_aliases,
            relation_synonyms=self.relation_synonyms,
        )

    def test_entity_alias_mapping(self):
        """Verify alias map resolution and whitespace trimming."""
        self.assertEqual(self.entity_norm.normalize_name("  usa  "), "United States")
        self.assertEqual(self.entity_norm.normalize_name("US"), "United States")
        self.assertEqual(self.entity_norm.normalize_name("elon"), "Elon Musk")

    def test_entity_casing_preservation(self):
        """Verify that acronyms and camelCase are preserved by default."""
        self.assertEqual(self.entity_norm.normalize_name("NASA"), "NASA")
        self.assertEqual(self.entity_norm.normalize_name("iPhone"), "iPhone")
        self.assertEqual(self.entity_norm.normalize_name("OpenAI"), "OpenAI")

    def test_entity_auto_titlecase(self):
        """Verify that auto_titlecase=True applies title casing."""
        norm = EntityNormalizer(auto_titlecase=True)
        self.assertEqual(norm.normalize_name("albert einstein"), "Albert Einstein")

    def test_cross_chunk_canonicalization(self):
        """Verify that entities are canonicalized based on first seen occurrence."""
        # Simulating first chunk discovery
        self.entity_norm.register_entity("OpenAI")
        
        # Second chunk mention with different casing
        self.assertEqual(self.entity_norm.normalize_name("openai"), "OpenAI")
        self.assertEqual(self.entity_norm.normalize_name("OPENAI"), "OpenAI")
        
        # New entity discovery
        self.entity_norm.register_entity("Microsoft")
        self.assertEqual(self.entity_norm.normalize_name("microsoft"), "Microsoft")

    def test_discover_entities(self):
        """Verify discover_entities populates the registry correctly."""
        entities = [
            Entity(name="DeepMind", entity_type=EntityType.ORGANIZATION),
            Entity(name="Google", entity_type=EntityType.ORGANIZATION),
        ]
        self.entity_norm.discover_entities(entities)
        self.assertEqual(self.entity_norm.normalize_name("deepmind"), "DeepMind")
        self.assertEqual(self.entity_norm.normalize_name("google"), "Google")

    def test_entity_object_normalization(self):
        """Verify normalization preserves entity type and metadata."""
        entity = Entity(name="tesla motors", entity_type=EntityType.ORGANIZATION, metadata={"source": "doc1"})
        norm_entity = self.entity_norm.normalize_entity(entity)
        self.assertEqual(norm_entity.name, "Tesla")
        self.assertEqual(norm_entity.entity_type, EntityType.ORGANIZATION)
        self.assertEqual(norm_entity.metadata["source"], "doc1")

    def test_relation_synonym_mapping(self):
        """Verify relation synonym map resolution, including defaults."""
        self.assertEqual(self.relation_norm.normalize_predicate("works at"), "EMPLOYED_BY")
        self.assertEqual(self.relation_norm.normalize_predicate("  is employed by  "), "EMPLOYED_BY")
        self.assertEqual(self.relation_norm.normalize_predicate("located in"), "LOCATED_IN")
        self.assertEqual(self.relation_norm.normalize_predicate("is part of"), "PART_OF")
        self.assertEqual(self.relation_norm.normalize_predicate("ceo of"), "LEADS")

    def test_relation_snake_case_fallback(self):
        """Verify unmapped relations are converted to UPPER_SNAKE_CASE."""
        self.assertEqual(self.relation_norm.normalize_predicate("invested in"), "INVESTED_IN")
        self.assertEqual(self.relation_norm.normalize_predicate("headquartered-in"), "HEADQUARTERED_IN")

    def test_schema_normalizer_single_triplet(self):
        """Verify end-to-end normalization of a full Triplet."""
        subj = Entity(name="elon", entity_type=EntityType.PERSON)
        rel = Relation(predicate="ceo of")
        obj = Entity(name="tesla motors", entity_type=EntityType.ORGANIZATION)
        triplet = Triplet(subject=subj, predicate=rel, object=obj, confidence=0.95)

        norm_triplet = self.schema_norm.normalize_triplet(triplet)
        self.assertEqual(norm_triplet.subject.name, "Elon Musk")
        self.assertEqual(norm_triplet.predicate.predicate, "LEADS")
        self.assertEqual(norm_triplet.object.name, "Tesla")
        self.assertEqual(norm_triplet.confidence, 0.95)

    def test_schema_normalizer_batch(self):
        """Verify batch normalization across multiple triplets."""
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
        batch = self.schema_norm.normalize_batch([t1, t2])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].subject.name, "Elon Musk")
        self.assertEqual(batch[0].object.name, "SpaceX")
        self.assertEqual(batch[1].predicate.predicate, "EMPLOYED_BY")
        self.assertEqual(batch[1].subject.name, "Tim Cook")

    def test_relation_linguistic_variations(self):
        """Verify linguistic variations map to canonical relations."""
        self.assertEqual(self.relation_norm.normalize_predicate("is a part of"), "PART_OF")
        self.assertEqual(self.relation_norm.normalize_predicate("is a member of"), "MEMBER_OF")
        self.assertEqual(self.relation_norm.normalize_predicate("member of"), "MEMBER_OF")

    def test_two_pass_cross_chunk_normalization(self):
        """Verify prepare_registry warms up cross-chunk entity names for subsequent normalization."""
        norm = SchemaNormalizer()
        # Chunk 1 has well-cased entities
        chunk1_triplets = [
            Triplet(
                subject=Entity(name="OpenAI"),
                predicate=Relation(predicate="developed"),
                object=Entity(name="ChatGPT"),
            )
        ]
        # Chunk 2 has lowercase mentions
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
        self.assertEqual(normalized_chunk2[0].subject.name, "OpenAI")
        self.assertEqual(normalized_chunk2[0].object.name, "ChatGPT")

    def test_schema_validation_constraints(self):
        """Verify validation errors on invalid entities, relations, or confidences."""
        with self.assertRaises(ValueError):
            Entity(name="")
        with self.assertRaises(ValueError):
            Entity(name="   ")
        with self.assertRaises(ValueError):
            Relation(predicate="")
        with self.assertRaises(ValueError):
            Triplet(
                subject=Entity(name="Valid"),
                predicate=Relation(predicate="VALID"),
                object=Entity(name="Valid"),
                confidence=1.5,
            )

if __name__ == "__main__":
    unittest.main()

