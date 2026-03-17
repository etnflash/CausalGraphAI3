"""RAG retriever — combines document chunk and ontology concept retrieval."""

import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from retrieval.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Aggregated retrieval result for a single query."""

    query: str
    document_chunks: list[VectorRecord]
    ontology_concepts: list[VectorRecord]

    def to_context_string(self) -> str:
        """Serialise both retrieval sets into a prompt-ready context block."""
        lines: list[str] = ["=== Relevant Document Passages ==="]
        for i, chunk in enumerate(self.document_chunks, 1):
            lines.append(f"[{i}] {chunk.text}")

        lines.append("\n=== Relevant ESG Ontology Concepts ===")
        for i, concept in enumerate(self.ontology_concepts, 1):
            meta = concept.metadata
            label = meta.get("label", "")
            definition = meta.get("definition", "")
            lines.append(f"[{i}] {label}: {definition}")

        return "\n".join(lines)


class Retriever:
    """Perform top-k retrieval of document chunks and ontology concepts.

    Args:
        vector_store: Shared :class:`VectorStore` instance.
        embedding_fn: Callable ``(texts: list[str]) -> list[list[float]]``.
        top_k_documents: Number of document chunks to retrieve.
        top_k_ontology: Number of ontology concepts to retrieve.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_fn,
        top_k_documents: int | None = None,
        top_k_ontology: int | None = None,
    ) -> None:
        self._store = vector_store
        self._embed = embedding_fn
        self._top_k_docs = top_k_documents or settings.top_k_documents
        self._top_k_onto = top_k_ontology or settings.top_k_ontology

    def retrieve(self, query: str) -> RetrievalResult:
        """Embed *query* and retrieve relevant chunks + ontology concepts.

        Args:
            query: Natural-language question or search string.

        Returns:
            :class:`RetrievalResult` with matched document chunks and
            ontology concepts.
        """
        logger.info("Retrieving context for query: '%s'", query[:80])

        query_vector = self._embed([query])[0]

        doc_chunks = self._store.search(
            query_vector=query_vector,
            top_k=self._top_k_docs,
            namespace="document",
        )
        onto_concepts = self._store.search(
            query_vector=query_vector,
            top_k=self._top_k_onto,
            namespace="ontology",
        )

        logger.info(
            "Retrieved %d document chunks and %d ontology concepts.",
            len(doc_chunks),
            len(onto_concepts),
        )

        return RetrievalResult(
            query=query,
            document_chunks=doc_chunks,
            ontology_concepts=onto_concepts,
        )
