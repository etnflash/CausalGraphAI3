"""Graph builder — persists extracted ESG entities and relations in Neo4j."""

import logging
from typing import Any

from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Cypher templates ─────────────────────────────────────────────────────────

_MERGE_ENTITY = """\
MERGE (e:ESGEntity {name: $name})
ON CREATE SET
    e.ontology_id    = $ontology_id,
    e.ontology_label = $ontology_label,
    e.category       = $category,
    e.created_at     = timestamp()
ON MATCH SET
    e.ontology_id    = coalesce($ontology_id,    e.ontology_id),
    e.ontology_label = coalesce($ontology_label, e.ontology_label),
    e.category       = coalesce($category,       e.category),
    e.updated_at     = timestamp()
RETURN e
"""

_MERGE_RELATION = """\
MATCH (s:ESGEntity {name: $subject})
MATCH (o:ESGEntity {name: $object})
MERGE (s)-[r:{predicate} {{type: $type}}]->(o)
ON CREATE SET
    r.value      = $value,
    r.created_at = timestamp()
ON MATCH SET
    r.value      = coalesce($value, r.value),
    r.updated_at = timestamp()
RETURN r
"""


def build_graph(
    extraction_result: dict[str, Any],
    client: Neo4jClient,
    document_id: str | None = None,
) -> dict[str, int]:
    """Persist extracted entities and relations in the Neo4j knowledge graph.

    Calls :func:`upsert_entities` and :func:`upsert_relations` in sequence.

    Args:
        extraction_result: Dict with ``"entities"`` and ``"relations"`` lists,
                           as returned by :func:`extraction.extractor.extract_entities_relations`.
        client: Connected :class:`Neo4jClient` instance.
        document_id: Optional source document identifier attached as metadata.

    Returns:
        Dict with ``"entities_upserted"`` and ``"relations_upserted"`` counts.
    """
    entities_upserted = upsert_entities(
        extraction_result.get("entities", []), client, document_id=document_id
    )
    relations_upserted = upsert_relations(
        extraction_result.get("relations", []), client
    )

    logger.info(
        "Graph build complete — %d entities, %d relations.",
        entities_upserted,
        relations_upserted,
    )
    return {
        "entities_upserted": entities_upserted,
        "relations_upserted": relations_upserted,
    }


def upsert_entities(
    entities: list[dict[str, Any]],
    client: Neo4jClient,
    document_id: str | None = None,
) -> int:
    """Merge ESG entity nodes into the graph.

    Args:
        entities: List of entity dicts (keys: ``name``, ``ontology_id``,
                  ``ontology_label``, ``category``).
        client: Active :class:`Neo4jClient`.
        document_id: Optional source document ID attached to each node.

    Returns:
        Number of entities processed.
    """
    count = 0
    for entity in entities:
        name = entity.get("name", "").strip()
        if not name:
            continue
        params: dict[str, Any] = {
            "name": name,
            "ontology_id": entity.get("ontology_id"),
            "ontology_label": entity.get("ontology_label"),
            "category": entity.get("category"),
        }
        if document_id:
            params["document_id"] = document_id
        try:
            client.run_query(_MERGE_ENTITY, params)
            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to upsert entity '%s': %s", name, exc)
    return count


def upsert_relations(
    relations: list[dict[str, Any]],
    client: Neo4jClient,
) -> int:
    """Merge ESG relation edges into the graph.

    Args:
        relations: List of relation dicts (keys: ``subject``, ``predicate``,
                   ``object``, ``type``, ``value``).
        client: Active :class:`Neo4jClient`.

    Returns:
        Number of relations processed.
    """
    count = 0
    for relation in relations:
        subject = relation.get("subject", "").strip()
        predicate = relation.get("predicate", "").strip().upper()
        obj = relation.get("object", "").strip()

        if not (subject and predicate and obj):
            continue

        # Neo4j relationship types must be uppercase identifiers
        safe_predicate = predicate.replace(" ", "_")

        params: dict[str, Any] = {
            "subject": subject,
            "object": obj,
            "type": relation.get("type", "qualitative"),
            "value": relation.get("value"),
        }
        # Dynamic relationship type requires string interpolation in Cypher
        cypher = _MERGE_RELATION.format(predicate=safe_predicate)
        try:
            client.run_query(cypher, params)
            count += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to upsert relation '%s -[%s]-> %s': %s",
                subject, predicate, obj, exc,
            )
    return count
