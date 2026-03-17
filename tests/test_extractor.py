"""Unit tests for the LLM extraction helper (no real LLM calls)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.extractor import ALLOWED_RELATIONS, _parse_llm_response


class TestParseLlmResponse:
    def test_valid_json(self):
        raw = """{
  "entities": [{"name": "CO2", "ontology_id": "esg:GHGEmissions", "ontology_label": "GHG Emissions", "category": "Environmental"}],
  "relations": [{"subject": "CO2", "predicate": "increases", "object": "temperature", "type": "quantitative", "value": "2°C"}]
}"""
        result = _parse_llm_response(raw)
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1
        assert result["relations"][0]["predicate"] == "increases"

    def test_markdown_code_fence_stripped(self):
        raw = "```json\n{\"entities\": [], \"relations\": []}\n```"
        result = _parse_llm_response(raw)
        assert result["entities"] == []
        assert result["relations"] == []

    def test_disallowed_relation_filtered(self):
        raw = """{
  "entities": [],
  "relations": [
    {"subject": "a", "predicate": "hacks", "object": "b", "type": "qualitative", "value": null}
  ]
}"""
        result = _parse_llm_response(raw)
        assert result["relations"] == []

    def test_mixed_allowed_and_disallowed(self):
        raw = """{
  "entities": [],
  "relations": [
    {"subject": "a", "predicate": "causes", "object": "b", "type": "qualitative", "value": null},
    {"subject": "c", "predicate": "bad_verb", "object": "d", "type": "qualitative", "value": null}
  ]
}"""
        result = _parse_llm_response(raw)
        assert len(result["relations"]) == 1
        assert result["relations"][0]["predicate"] == "causes"

    def test_empty_response(self):
        result = _parse_llm_response("")
        assert result == {"entities": [], "relations": []}

    def test_no_json_object(self):
        result = _parse_llm_response("Sorry, I could not extract anything.")
        assert result == {"entities": [], "relations": []}

    def test_defaults_added_when_missing_keys(self):
        result = _parse_llm_response('{"entities": []}')
        assert "relations" in result

    def test_allowed_relations_set(self):
        assert "causes" in ALLOWED_RELATIONS
        assert "contributes_to" in ALLOWED_RELATIONS
        assert "increases" in ALLOWED_RELATIONS
        assert "reduces" in ALLOWED_RELATIONS
        assert "improves" in ALLOWED_RELATIONS
        assert "worsens" in ALLOWED_RELATIONS
