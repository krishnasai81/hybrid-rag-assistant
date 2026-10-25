
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import RAW_DIR, PROCESSED_DIR
from app.ingestion import (
    SUPPORTED_EXTENSIONS,
    ingest_file,
    upload_original_document,
)
from app.indexing import index_document_chunks
from app.retrieval import clear_retrieval_caches
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    UploadResponse,
)
from app.supabase_client import get_supabase_client, SUPABASE_BUCKET

router = APIRouter()
MAX_FILE_SIZE = 20 * 1024 * 1024


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        application="Hybrid RAG Assistant",
    )


def process_uploaded_document(path: Path) -> int:
    registered = None

    try:
        registered = upload_original_document(path)
        document_id = registered["document_id"]

        chunks = ingest_file(path, document_id=document_id)
        if not chunks:
            raise ValueError(
                "No readable text was found in the uploaded document."
            )

        index_document_chunks(document_id, chunks)
        clear_retrieval_caches()
        return len(chunks)

    except Exception:
        # Remove the document record (and its chunks) if processing fails.
        if registered:
            try:
                client = get_supabase_client()
                client.table("documents").delete().eq(
                    "id", registered["document_id"]
                ).execute()

                client.storage.from_(SUPABASE_BUCKET).remove(
                    [registered["storage_path"]]
                )
            except Exception as cleanup_error:
                print(f"Supabase cleanup failed: {cleanup_error}")

        raise


@router.post("/documents/upload", response_model=UploadResponse)
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

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        stored_name = f"{uuid4().hex}_{original_name}"
        stored_path = RAW_DIR / stored_name
        processed_path = PROCESSED_DIR / f"{Path(stored_name).stem}.json"
        stored_path.write_bytes(content)

        try:
            chunk_count = await run_in_threadpool(
                process_uploaded_document,
                stored_path,
            )
        except ValueError as exc:
            processed_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            print(f"Document processing failed: {exc}")
            raise HTTPException(
                status_code=500,
                detail=(
                    "Document processing or indexing failed. "
                    "Check the backend terminal for details."
                ),
            ) from exc
        finally:
            stored_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

        return UploadResponse(
            filename=original_name,
            chunks_created=chunk_count,
            message="Document stored and indexed successfully.",
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
        from app.generation import ask_documents

        result = await run_in_threadpool(ask_documents, question)

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources", []),
            verification=result.get("verification", {}),
        )
    except Exception as exc:
        print(f"Chat processing failed: {exc}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to answer the question. Check Supabase, "
                "the Groq API key, and the backend terminal."
            ),
        ) from exc
