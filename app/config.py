"""Application configuration loaded from environment variables or a .env file."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Google Cloud / Vertex AI ──────────────────────────────────────────────
    gcp_project: str = "your-gcp-project"
    gcp_location: str = "us-central1"
    vertex_embedding_model: str = "textembedding-gecko@003"
    vertex_llm_model: str = "gemini-1.0-pro"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector store ─────────────────────────────────────────────────────────
    # Supported values: "in_memory"  (swap for "faiss", "pinecone", etc.)
    vector_store_type: str = "in_memory"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    top_k_documents: int = 5
    top_k_ontology: int = 3

    # ── Ontology ──────────────────────────────────────────────────────────────
    ontology_path: str = "ontology/esg_ontology.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
