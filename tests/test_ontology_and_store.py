"""Unit tests for ontology loading and the vector store."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.ontology_loader import Ontology, OntologyConcept, load_ontology
from retrieval.vector_store import VectorRecord, VectorStore


# ── Ontology loader tests ─────────────────────────────────────────────────────

ONTOLOGY_PATH = Path(__file__).parent.parent / "ontology" / "esg_ontology.json"


class TestLoadOntology:
    def test_loads_bundled_ontology(self):
        onto = load_ontology(ONTOLOGY_PATH)
        assert isinstance(onto, Ontology)
        assert len(onto.concepts) > 0
        assert len(onto.relations) > 0

    def test_concepts_have_required_fields(self):
        onto = load_ontology(ONTOLOGY_PATH)
        for concept in onto.concepts:
            assert concept.id.startswith("esg:")
            assert concept.label
            assert concept.category in ("Environmental", "Social", "Governance")

    def test_get_concept_by_id(self):
        onto = load_ontology(ONTOLOGY_PATH)
        concept = onto.get_concept_by_id("esg:GHGEmissions")
        assert concept is not None
        assert concept.label == "Greenhouse Gas Emissions"

    def test_get_concept_by_id_missing(self):
        onto = load_ontology(ONTOLOGY_PATH)
        assert onto.get_concept_by_id("esg:NonExistent") is None

    def test_get_concepts_by_category(self):
        onto = load_ontology(ONTOLOGY_PATH)
        env = onto.get_concepts_by_category("Environmental")
        assert all(c.category == "Environmental" for c in env)
        assert len(env) > 0

    def test_to_embedding_text_includes_label(self):
        onto = load_ontology(ONTOLOGY_PATH)
        for concept in onto.concepts:
            embed_text = concept.to_embedding_text()
            assert concept.label in embed_text

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_ontology(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"no_concepts": true}')
        with pytest.raises(ValueError, match="concepts"):
            load_ontology(bad)


# ── Vector store tests ────────────────────────────────────────────────────────


class TestVectorStore:
    def setup_method(self):
        self.store = VectorStore()

    def test_upsert_and_len(self):
        self.store.upsert([1.0, 0.0], "doc A", "id1")
        assert len(self.store) == 1

    def test_upsert_generates_id(self):
        doc_id = self.store.upsert([1.0, 0.0], "doc A")
        assert isinstance(doc_id, str) and len(doc_id) > 0

    def test_search_returns_closest(self):
        self.store.upsert([1.0, 0.0, 0.0], "doc A", "a")
        self.store.upsert([0.0, 1.0, 0.0], "doc B", "b")
        self.store.upsert([0.0, 0.0, 1.0], "doc C", "c")
        results = self.store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0].doc_id == "a"

    def test_search_top_k_limit(self):
        for i in range(10):
            self.store.upsert([float(i), 0.0], f"doc {i}", str(i))
        results = self.store.search([1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_namespace_filter(self):
        self.store.upsert([1.0, 0.0], "doc", "d1", {"namespace": "document"})
        self.store.upsert([1.0, 0.0], "onto", "o1", {"namespace": "ontology"})
        doc_results = self.store.search([1.0, 0.0], namespace="document")
        assert all(r.metadata.get("namespace") == "document" for r in doc_results)
        assert len(doc_results) == 1

    def test_delete(self):
        self.store.upsert([1.0, 0.0], "doc", "id1")
        removed = self.store.delete("id1")
        assert removed is True
        assert len(self.store) == 0

    def test_delete_missing_returns_false(self):
        assert self.store.delete("nonexistent") is False

    def test_clear_all(self):
        self.store.upsert([1.0, 0.0], "doc A", "a")
        self.store.upsert([0.0, 1.0], "doc B", "b")
        count = self.store.clear()
        assert count == 2
        assert len(self.store) == 0

    def test_clear_by_namespace(self):
        self.store.upsert([1.0, 0.0], "doc", "d1", {"namespace": "document"})
        self.store.upsert([1.0, 0.0], "onto", "o1", {"namespace": "ontology"})
        removed = self.store.clear(namespace="document")
        assert removed == 1
        assert len(self.store) == 1

    def test_search_empty_store(self):
        results = self.store.search([1.0, 0.0])
        assert results == []
