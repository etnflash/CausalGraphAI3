"""FastAPI routes — document upload, ontology management, and RAG queries."""

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.config import settings
from embedding.vertex_embedding import embed_texts
from extraction.extractor import extract_entities_relations
from graph.graph_builder import build_graph
from graph.neo4j_client import Neo4jClient
from ingestion.chunker import chunk_text
from ingestion.document_loader import load_document_from_bytes
from ontology.ontology_embedder import embed_ontology as _embed_ontology
from ontology.ontology_loader import Ontology, load_ontology
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared singletons (swappable for dependency injection) ────────────────────

_vector_store = VectorStore()
_ontology: Ontology | None = None
_neo4j_client = Neo4jClient()


def get_vector_store() -> VectorStore:
    return _vector_store


def get_ontology() -> Ontology | None:
    return _ontology


def get_neo4j_client() -> Neo4jClient:
    return _neo4j_client


# ── Request / Response models ─────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    store_in_graph: bool = False


class QueryResponse(BaseModel):
    query: str
    context: str
    extraction: dict[str, Any]
    graph_result: dict[str, int] | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_stored: int


class OntologyResponse(BaseModel):
    concepts_loaded: int
    concepts_embedded: int
    version: str


class HealthResponse(BaseModel):
    status: str
    vector_store_size: int
    neo4j_connected: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check(
    store: Annotated[VectorStore, Depends(get_vector_store)],
    neo4j: Annotated[Neo4jClient, Depends(get_neo4j_client)],
) -> HealthResponse:
    """Return service health and basic stats."""
    return HealthResponse(
        status="ok",
        vector_store_size=len(store),
        neo4j_connected=neo4j.verify_connectivity(),
    )


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF or TXT ESG report")],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> UploadResponse:
    """Upload an ESG document, extract text, chunk, embed, and store.

    Accepted formats: **PDF**, **TXT**.

    Returns the number of chunks stored in the vector store.
    """
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    allowed_suffixes = {".pdf", ".txt"}
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Accepted: PDF, TXT.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    document_id = str(uuid.uuid4())

    try:
        text = load_document_from_bytes(file_bytes, file.filename)
    except Exception as exc:
        logger.exception("Text extraction failed for '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text extraction failed: {exc}",
        ) from exc

    chunks = chunk_text(text, metadata={"document_id": document_id, "filename": file.filename})
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the document.",
        )

    chunk_texts = [c.text for c in chunks]
    try:
        vectors = embed_texts(chunk_texts)
    except Exception as exc:
        logger.exception("Embedding failed for document '%s'", document_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding service error: {exc}",
        ) from exc

    for chunk, vector in zip(chunks, vectors):
        store.upsert(
            vector=vector,
            text=chunk.text,
            doc_id=f"{document_id}::chunk::{chunk.index}",
            metadata={
                "namespace": "document",
                "document_id": document_id,
                "filename": file.filename,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            },
        )

    logger.info(
        "Stored %d chunks for document '%s' (id=%s).",
        len(chunks),
        file.filename,
        document_id,
    )
    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_stored=len(chunks),
    )


@router.post("/ontology/load", response_model=OntologyResponse, tags=["ontology"])
async def load_and_embed_ontology(
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> OntologyResponse:
    """Load the ESG ontology from the configured JSON file and embed it.

    Idempotent — calling it multiple times refreshes the ontology embeddings.
    """
    global _ontology  # noqa: PLW0603

    try:
        ontology = load_ontology(settings.ontology_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    _ontology = ontology

    # Remove stale ontology embeddings before reinserting
    store.clear(namespace="ontology")

    try:
        embedded_count = _embed_ontology(ontology, store, embed_texts)
    except Exception as exc:
        logger.exception("Ontology embedding failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding service error: {exc}",
        ) from exc

    return OntologyResponse(
        concepts_loaded=len(ontology.concepts),
        concepts_embedded=embedded_count,
        version=ontology.version,
    )


@router.post("/query", response_model=QueryResponse, tags=["retrieval"])
async def query_system(
    request: QueryRequest,
    store: Annotated[VectorStore, Depends(get_vector_store)],
    neo4j: Annotated[Neo4jClient, Depends(get_neo4j_client)],
) -> QueryResponse:
    """Run a RAG query: retrieve context, extract entities/relations, optionally store in Neo4j.

    Steps:
    1. Embed the query.
    2. Retrieve top-k document chunks + ontology concepts.
    3. Call the LLM to extract entities and relations.
    4. Optionally persist the extraction to Neo4j.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty.",
        )

    retriever = Retriever(
        vector_store=store,
        embedding_fn=embed_texts,
    )

    try:
        retrieval_result = retriever.retrieve(request.query)
    except Exception as exc:
        logger.exception("Retrieval failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Retrieval error: {exc}",
        ) from exc

    context = retrieval_result.to_context_string()

    try:
        extraction = extract_entities_relations(
            text="\n".join(c.text for c in retrieval_result.document_chunks),
            ontology_context="\n".join(c.text for c in retrieval_result.ontology_concepts),
        )
    except Exception as exc:
        logger.exception("LLM extraction failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM extraction error: {exc}",
        ) from exc

    graph_result = None
    if request.store_in_graph:
        try:
            graph_result = build_graph(extraction, neo4j)
        except Exception as exc:
            logger.warning("Graph storage failed (non-fatal): %s", exc)
            graph_result = {"entities_upserted": 0, "relations_upserted": 0}

    return QueryResponse(
        query=request.query,
        context=context,
        extraction=extraction,
        graph_result=graph_result,
    )
