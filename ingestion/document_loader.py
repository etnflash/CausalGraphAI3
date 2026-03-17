"""Document loader — extracts plain text from PDF and TXT uploads."""

import io
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_document(file_path: Union[str, Path]) -> str:
    """Load a document from *file_path* and return its full text.

    Args:
        file_path: Absolute or relative path to a PDF or TXT file.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If *file_path* does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    raw_bytes = path.read_bytes()
    return load_document_from_bytes(raw_bytes, path.name)


def load_document_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from *file_bytes* using the extension in *filename*.

    Args:
        file_bytes: Raw binary content of the file.
        filename: Original filename (used to detect extension).

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _extract_text_from_pdf(file_bytes)
    elif suffix == ".txt":
        text = _extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Extracted %d characters from '%s'", len(text), filename)
    return text


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file using PyPDF2.

    Args:
        file_bytes: Raw PDF binary content.

    Returns:
        Concatenated plain text from all pages.
    """
    try:
        import PyPDF2  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyPDF2 is required for PDF extraction. "
            "Install it with: pip install PyPDF2"
        ) from exc

    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def _extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode *file_bytes* as UTF-8 text (falls back to latin-1).

    Args:
        file_bytes: Raw text file binary content.

    Returns:
        Decoded string.
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")
