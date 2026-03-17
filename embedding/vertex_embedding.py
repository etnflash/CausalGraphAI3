"""Vertex AI embedding client — wraps the Vertex AI text embedding API."""

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-initialised Vertex AI client (avoids import-time side effects)
_vertex_client: Any = None


def _get_client():
    """Return a lazily initialised Vertex AI TextEmbeddingModel instance."""
    global _vertex_client  # noqa: PLW0603
    if _vertex_client is None:
        try:
            import vertexai  # noqa: PLC0415
            from vertexai.language_models import TextEmbeddingModel  # noqa: PLC0415

            vertexai.init(
                project=settings.gcp_project,
                location=settings.gcp_location,
            )
            _vertex_client = TextEmbeddingModel.from_pretrained(
                settings.vertex_embedding_model
            )
            logger.info(
                "Vertex AI embedding model '%s' initialised.",
                settings.vertex_embedding_model,
            )
        except ImportError as exc:
            raise ImportError(
                "google-cloud-aiplatform is required. "
                "Install it with: pip install google-cloud-aiplatform"
            ) from exc
    return _vertex_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings.

    Calls the Vertex AI text embedding API in batches of up to 250 texts
    (the per-request limit for ``textembedding-gecko``).

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of embedding vectors (one per input text), where each vector
        is a ``list[float]`` of length equal to the model's output dimension.

    Raises:
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list.")

    model = _get_client()

    batch_size = 250
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings])
        logger.debug("Embedded batch %d/%d.", i // batch_size + 1, -(-len(texts) // batch_size))

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string.

    Args:
        query: The query text.

    Returns:
        A single embedding vector as ``list[float]``.
    """
    return embed_texts([query])[0]
