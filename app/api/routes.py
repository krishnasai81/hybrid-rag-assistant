
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import RAW_DIR, PROCESSED_DIR
from app.ingestion import SUPPORTED_EXTENSIONS, ingest_file
from app.indexing import build_indexes
from app.retrieval import clear_retrieval_caches
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    UploadResponse,
)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        application="Hybrid RAG Assistant",
    )


def process_uploaded_document(path: Path) -> int:
    chunks = ingest_file(path)

    if not chunks:
        raise ValueError(
            "No readable text was found in the uploaded document."
        )

    # Rebuild ChromaDB and BM25 indexes from processed documents.
    build_indexes()

    # Make subsequent questions use the newly built indexes.
    clear_retrieval_caches()

    return len(chunks)


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    original_name = Path(file.filename).name
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Supported formats: PDF, TXT, MD, HTML.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    try:
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 20 MB.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        stored_name = f"{uuid4().hex}_{original_name}"
        stored_path = RAW_DIR / stored_name
        processed_path = (
            PROCESSED_DIR / f"{Path(stored_name).stem}.json"
        )

        stored_path.write_bytes(content)

        try:
            chunk_count = await run_in_threadpool(
                process_uploaded_document,
                stored_path,
            )
        except ValueError as exc:
            stored_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            # Keep the underlying error in the backend terminal.
            print(f"Document processing failed: {exc}")

            raise HTTPException(
                status_code=500,
                detail=(
                    "Document processing or indexing failed. "
                    "Check the backend terminal for details."
                ),
            ) from exc

        return UploadResponse(
            filename=original_name,
            chunks_created=chunk_count,
            message="Document processed and indexes rebuilt successfully.",
        )
    finally:
        await file.close()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot contain only whitespace.",
        )

    try:
        # Import here so the API health endpoint can still start
        # even if the Groq API key has not been configured.
        from app.generation import ask_documents

        result = await run_in_threadpool(
            ask_documents,
            question,
        )

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources", []),
            verification=result.get("verification", {}),
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Chat processing failed: {exc}")

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to answer the question. Check that documents "
                "have been indexed, your Groq API key is configured, "
                "and the backend terminal has no errors."
            ),
        ) from exc
    