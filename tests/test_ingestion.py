"""Unit tests for document loading and text chunking."""

import sys
from pathlib import Path

import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    _extract_text_from_txt,
    load_document_from_bytes,
)
from ingestion.chunker import TextChunk, chunk_text


# ── Document loader tests ─────────────────────────────────────────────────────


class TestLoadDocumentFromBytes:
    def test_txt_utf8(self):
        text = load_document_from_bytes(b"Hello ESG world!", "report.txt")
        assert text == "Hello ESG world!"

    def test_txt_latin1_fallback(self):
        # latin-1 encoded bytes that are invalid UTF-8
        content = "résumé".encode("latin-1")
        text = load_document_from_bytes(content, "report.txt")
        assert "sum" in text  # "résumé" decoded via latin-1

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document_from_bytes(b"data", "document.docx")

    def test_pdf_extraction(self):
        """Minimal smoke-test for PDF extraction using PyPDF2."""
        try:
            import PyPDF2
            import io

            # Create a tiny valid PDF in memory
            writer = PyPDF2.PdfWriter()
            writer.add_blank_page(width=72, height=72)
            buf = io.BytesIO()
            writer.write(buf)
            pdf_bytes = buf.getvalue()

            # Should not raise
            result = load_document_from_bytes(pdf_bytes, "blank.pdf")
            assert isinstance(result, str)
        except ImportError:
            pytest.skip("PyPDF2 not installed")


# ── Chunker tests ─────────────────────────────────────────────────────────────


class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_long_text_multiple_chunks(self):
        # ~1500 characters → should produce multiple 512-char chunks
        text = "word " * 300  # 1500 chars
        chunks = chunk_text(text, chunk_size=512, chunk_overlap=64)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self):
        text = "x " * 500
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_metadata_propagated(self):
        text = "word " * 100
        meta = {"document_id": "doc-1", "filename": "test.txt"}
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10, metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["document_id"] == "doc-1"
            assert chunk.metadata["filename"] == "test.txt"

    def test_overlap_invalid_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("some text", chunk_size=50, chunk_overlap=50)

    def test_chunk_is_dataclass(self):
        chunks = chunk_text("hello world", chunk_size=512, chunk_overlap=64)
        assert isinstance(chunks[0], TextChunk)
        assert chunks[0].start_char == 0
