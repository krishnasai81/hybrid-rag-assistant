
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Hybrid RAG Assistant API",
    description=(
        "Backend API for document ingestion, "
        "hybrid retrieval, and grounded question answering."
    ),
    version="0.1.0",
)

# Allow the local React development server to access the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Hybrid RAG Assistant API is running.",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
        