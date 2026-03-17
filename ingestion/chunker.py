"""Text chunker — splits documents into overlapping fixed-size chunks."""

import logging
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict | None = None,
) -> list[TextChunk]:
    """Split *text* into overlapping chunks of *chunk_size* characters.

    Uses a simple character-based sliding window with *chunk_overlap*
    characters of overlap between consecutive chunks.  Chunks are split
    on whitespace boundaries where possible to avoid cutting mid-word.

    Args:
        text: Full document text.
        chunk_size: Maximum number of characters per chunk.
                    Defaults to ``settings.chunk_size``.
        chunk_overlap: Number of characters shared between consecutive chunks.
                       Defaults to ``settings.chunk_overlap``.
        metadata: Optional key-value metadata attached to every chunk.

    Returns:
        List of :class:`TextChunk` objects in document order.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    text = text.strip()
    if not text:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break on a whitespace boundary to avoid splitting words
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                TextChunk(
                    text=chunk_text_str,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata=dict(metadata),
                )
            )
            index += 1

        start += step

    logger.info(
        "Chunked text (%d chars) into %d chunks "
        "(size=%d, overlap=%d)",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
