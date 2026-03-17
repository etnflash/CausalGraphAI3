"""Ontology loader — reads and normalises the ESG ontology JSON file."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OntologyConcept:
    """A single ESG ontology concept ready for embedding and retrieval."""

    id: str
    label: str
    category: str
    aliases: list[str] = field(default_factory=list)
    definition: str = ""
    unit_examples: list[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Return a rich text representation suitable for embedding.

        The text concatenates the label, definition, category, and aliases
        so that semantic search can match varied surface forms.
        """
        parts = [
            f"Concept: {self.label}",
            f"Category: {self.category}",
        ]
        if self.definition:
            parts.append(f"Definition: {self.definition}")
        if self.aliases:
            parts.append(f"Also known as: {', '.join(self.aliases)}")
        return ". ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "aliases": self.aliases,
            "definition": self.definition,
            "unit_examples": self.unit_examples,
        }


@dataclass
class Ontology:
    """Container for all ESG ontology concepts and allowed relation types."""

    concepts: list[OntologyConcept] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    version: str = ""
    description: str = ""

    def get_concept_by_id(self, concept_id: str) -> OntologyConcept | None:
        """Look up a concept by its unique identifier."""
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        return None

    def get_concepts_by_category(self, category: str) -> list[OntologyConcept]:
        """Return all concepts belonging to *category* (case-insensitive)."""
        category_lower = category.lower()
        return [c for c in self.concepts if c.category.lower() == category_lower]


def load_ontology(json_path: str | Path) -> Ontology:
    """Load the ESG ontology from a JSON file.

    Args:
        json_path: Path to the ontology JSON file.

    Returns:
        An :class:`Ontology` instance populated with concepts and relations.

    Raises:
        FileNotFoundError: If *json_path* does not exist.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {json_path}")

    with path.open(encoding="utf-8") as fh:
        data: dict = json.load(fh)

    if "concepts" not in data:
        raise ValueError("Ontology JSON must contain a 'concepts' key.")

    concepts = [
        OntologyConcept(
            id=item["id"],
            label=item["label"],
            category=item.get("category", ""),
            aliases=item.get("aliases", []),
            definition=item.get("definition", ""),
            unit_examples=item.get("unit_examples", []),
        )
        for item in data["concepts"]
    ]

    ontology = Ontology(
        concepts=concepts,
        relations=data.get("relations", []),
        version=data.get("version", ""),
        description=data.get("description", ""),
    )

    logger.info(
        "Loaded ontology v%s: %d concepts, %d relation types",
        ontology.version,
        len(ontology.concepts),
        len(ontology.relations),
    )
    return ontology
