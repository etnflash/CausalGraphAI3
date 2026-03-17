"""In-memory vector store backed by NumPy cosine similarity search.

This is the default swappable vector store implementation.  Replace this
module (or swap ``VECTOR_STORE_TYPE`` in config) to use FAISS, Pinecone,
Weaviate, etc. — all components depend only on the :class:`VectorStore`
interface below.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """A single stored record: text, vector, and metadata."""

    doc_id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Thread-safe in-memory vector store with cosine similarity search.

    Attributes:
        records: All stored :class:`VectorRecord` objects keyed by doc_id.
    """

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert(
        self,
        vector: list[float],
        text: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert or update a record.

        Args:
            vector: Embedding vector.
            text: Original text the vector was generated from.
            doc_id: Optional stable identifier.  A UUID is generated if omitted.
            metadata: Arbitrary key-value metadata stored alongside the vector.

        Returns:
            The ``doc_id`` of the upserted record.
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self._records[doc_id] = VectorRecord(
            doc_id=doc_id,
            text=text,
            vector=vector,
            metadata=metadata or {},
        )
        return doc_id

    def delete(self, doc_id: str) -> bool:
        """Remove a record by *doc_id*.  Returns ``True`` if it existed."""
        return self._records.pop(doc_id, None) is not None

    def clear(self, namespace: str | None = None) -> int:
        """Remove all records, optionally filtering by *namespace* metadata.

        Args:
            namespace: If provided, only records whose ``metadata["namespace"]``
                       equals this value are deleted.

        Returns:
            Number of records removed.
        """
        if namespace is None:
            count = len(self._records)
            self._records.clear()
            return count

        to_delete = [
            doc_id
            for doc_id, rec in self._records.items()
            if rec.metadata.get("namespace") == namespace
        ]
        for doc_id in to_delete:
            del self._records[doc_id]
        return len(to_delete)

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str | None = None,
    ) -> list[VectorRecord]:
        """Return the *top_k* most similar records to *query_vector*.

        Uses cosine similarity.  Optionally filters by ``metadata["namespace"]``.

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of results to return.
            namespace: If provided, search only within this namespace.

        Returns:
            List of :class:`VectorRecord` objects sorted by descending similarity.
        """
        candidates = list(self._records.values())
        if namespace is not None:
            candidates = [
                r for r in candidates if r.metadata.get("namespace") == namespace
            ]

        if not candidates:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scores: list[tuple[float, VectorRecord]] = []
        for record in candidates:
            v = np.array(record.vector, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q, v) / (q_norm * v_norm))
            scores.append((score, record))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scores[:top_k]]

    def __len__(self) -> int:
        return len(self._records)
