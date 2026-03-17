"""LLM-based entity and relation extractor using Vertex AI Gemini."""

import json
import logging
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Normalised relation vocabulary ───────────────────────────────────────────
ALLOWED_RELATIONS = {
    "causes",
    "contributes_to",
    "increases",
    "reduces",
    "improves",
    "worsens",
    "measures",
    "targets",
    "reports_on",
    "manages",
}

# ── Structured extraction prompt template ─────────────────────────────────────
_EXTRACTION_PROMPT = """\
You are an ESG knowledge graph extraction assistant.

Given the TEXT and the ESG ONTOLOGY CONTEXT below, extract:
1. ESG entities — map each entity to the closest ontology concept if possible.
2. Relations between entities — use ONLY the allowed relation vocabulary.
3. Classify each relation as "qualitative" or "quantitative".
   A relation is quantitative if numeric values or units are present.

## Allowed relation types
{allowed_relations}

## ESG Ontology Context
{ontology_context}

## Text
{text}

## Output format (strict JSON, no extra text)
{{
  "entities": [
    {{
      "name": "<surface form from text>",
      "ontology_id": "<esg:ConceptID or null>",
      "ontology_label": "<human-readable label or null>",
      "category": "<Environmental | Social | Governance | null>"
    }}
  ],
  "relations": [
    {{
      "subject": "<entity name>",
      "predicate": "<relation type from allowed list>",
      "object": "<entity name>",
      "type": "<qualitative | quantitative>",
      "value": "<numeric value with unit if quantitative, else null>"
    }}
  ]
}}
"""


def extract_entities_relations(
    text: str,
    ontology_context: str,
) -> dict[str, Any]:
    """Use the Vertex AI LLM to extract ESG entities and relations from *text*.

    Args:
        text: The document passage to analyse.
        ontology_context: Relevant ontology concepts serialised as a string
                          (typically from :meth:`RetrievalResult.to_context_string`).

    Returns:
        Parsed JSON dict with keys ``"entities"`` and ``"relations"``.
        Returns a fallback empty structure if the LLM response cannot be parsed.
    """
    prompt = _EXTRACTION_PROMPT.format(
        allowed_relations=", ".join(sorted(ALLOWED_RELATIONS)),
        ontology_context=ontology_context,
        text=text,
    )

    raw_response = _call_llm(prompt)
    return _parse_llm_response(raw_response)


def _call_llm(prompt: str) -> str:
    """Send *prompt* to the Vertex AI Gemini model and return the text response.

    Args:
        prompt: Full prompt string.

    Returns:
        Raw text output from the model.
    """
    try:
        import vertexai  # noqa: PLC0415
        from vertexai.generative_models import GenerativeModel  # noqa: PLC0415

        vertexai.init(
            project=settings.gcp_project,
            location=settings.gcp_location,
        )
        model = GenerativeModel(settings.vertex_llm_model)
        response = model.generate_content(prompt)
        return response.text
    except ImportError as exc:
        raise ImportError(
            "google-cloud-aiplatform is required. "
            "Install it with: pip install google-cloud-aiplatform"
        ) from exc


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Extract and validate JSON from the raw LLM response string.

    Args:
        raw: Raw text from the LLM, expected to contain a JSON block.

    Returns:
        Parsed dict with ``"entities"`` and ``"relations"`` lists.
        Returns an empty structure on any parse error.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Attempt to locate the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("No JSON object found in LLM response.")
        return {"entities": [], "relations": []}

    json_str = cleaned[start:end]
    try:
        result: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        return {"entities": [], "relations": []}

    # Normalise to expected structure
    result.setdefault("entities", [])
    result.setdefault("relations", [])

    # Filter relations to the allowed vocabulary
    result["relations"] = [
        r for r in result["relations"]
        if r.get("predicate") in ALLOWED_RELATIONS
    ]

    return result
