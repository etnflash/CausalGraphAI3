"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup: initialise shared resources (connections, caches, etc.)
    yield
    # Shutdown: release shared resources


app = FastAPI(
    title="ESG Knowledge Graph + RAG System",
    description=(
        "A scalable backend for ESG report analysis using knowledge graphs "
        "and Retrieval-Augmented Generation (RAG)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
