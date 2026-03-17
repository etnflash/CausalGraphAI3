"""Integration tests for FastAPI endpoints (no external services required)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ── Health check ──────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, client):
        # Neo4j is not running in tests; verify_connectivity returns False — that's fine
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "vector_store_size" in data
        assert "neo4j_connected" in data


# ── Document upload ───────────────────────────────────────────────────────────

_DUMMY_VECTOR = [0.1] * 768  # fake 768-dim embedding


class TestUploadDocument:
    def _fake_embed(self, texts):
        return [_DUMMY_VECTOR for _ in texts]

    def test_upload_txt(self, client):
        content = b"This is a test ESG report.\n" * 30
        with patch("api.routes.embed_texts", side_effect=self._fake_embed):
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("report.txt", content, "text/plain")},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "report.txt"
        assert data["chunks_stored"] >= 1
        assert len(data["document_id"]) > 0

    def test_upload_unsupported_type(self, client):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.docx", b"data", "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_empty_file(self, client):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("report.txt", b"", "text/plain")},
        )
        assert response.status_code == 400


# ── Ontology load ─────────────────────────────────────────────────────────────


class TestOntologyLoad:
    def _fake_embed(self, texts):
        return [[0.1] * 768 for _ in texts]

    def test_load_ontology_success(self, client):
        with patch("api.routes.embed_texts", side_effect=self._fake_embed):
            response = client.post("/api/v1/ontology/load")
        assert response.status_code == 200
        data = response.json()
        assert data["concepts_loaded"] > 0
        assert data["concepts_embedded"] > 0
        assert data["version"] != ""


# ── Query ─────────────────────────────────────────────────────────────────────


class TestQueryEndpoint:
    def _fake_embed(self, texts):
        return [[0.1] * 768 for _ in texts]

    def _fake_extract(self, text, ontology_context):
        return {
            "entities": [{"name": "CO2 emissions", "ontology_id": None, "ontology_label": None, "category": None}],
            "relations": [],
        }

    def test_empty_query_rejected(self, client):
        response = client.post("/api/v1/query", json={"query": "  "})
        assert response.status_code == 400

    def test_query_returns_structure(self, client):
        with (
            patch("api.routes.embed_texts", side_effect=self._fake_embed),
            patch("api.routes.extract_entities_relations", side_effect=self._fake_extract),
        ):
            response = client.post("/api/v1/query", json={"query": "What are GHG emissions?"})
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "context" in data
        assert "extraction" in data
        assert "entities" in data["extraction"]
        assert "relations" in data["extraction"]
