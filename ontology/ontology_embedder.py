"""Ontology embedder — generates and stores embeddings for ESG concepts."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ontology.ontology_loader import Ontology, OntologyConcept
    from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def embed_ontology(
    ontology: "Ontology",
    vector_store: "VectorStore",
    embedding_fn,
) -> int:
    """Embed all ontology concepts and persist them in *vector_store*.

    Each concept is converted to a rich text representation via
    :meth:`OntologyConcept.to_embedding_text`, then embedded using
    *embedding_fn* and stored with ``namespace="ontology"``.

    Args:
        ontology: Loaded :class:`Ontology` instance.
        vector_store: Target :class:`VectorStore` to upsert into.
        embedding_fn: Callable ``(texts: list[str]) -> list[list[float]]``
                      that returns one embedding vector per input text.

    Returns:
        Number of concepts embedded and stored.
    """
    if not ontology.concepts:
        logger.warning("Ontology contains no concepts; nothing to embed.")
        return 0

    texts = [concept.to_embedding_text() for concept in ontology.concepts]
    metadata_list = [concept.to_dict() for concept in ontology.concepts]
    ids = [concept.id for concept in ontology.concepts]

    logger.info("Embedding %d ontology concepts …", len(texts))
    vectors = embedding_fn(texts)

    for doc_id, vector, text, meta in zip(ids, vectors, texts, metadata_list):
        vector_store.upsert(
            doc_id=doc_id,
            vector=vector,
            text=text,
            metadata={**meta, "namespace": "ontology"},
        )

    logger.info("Stored %d ontology embeddings in vector store.", len(ids))
    return len(ids)
