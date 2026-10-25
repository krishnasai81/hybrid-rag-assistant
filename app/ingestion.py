
import hashlib
import json
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import (
    RAW_DIR,
    PROCESSED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.supabase_client import (
    get_supabase_client,
    SUPABASE_BUCKET,
)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}


def clean_text(text: str) -> str:
    """Normalize extracted text without removing meaningful content."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_document(path: Path) -> list[dict[str, Any]]:
    """Extract text blocks with source and page metadata."""
    extension = path.suffix.lower()
    records = []

    if extension == ".pdf":
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text"))
                if text:
                    records.append({
                        "text": text,
                        "source": path.name,
                        "page": page_number,
                    })

    elif extension in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = clean_text(soup.get_text(separator="\n"))
        if text:
            records.append({
                "text": text,
                "source": path.name,
                "page": None,
            })

    else:
        text = clean_text(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if text:
            records.append({
                "text": text,
                "source": path.name,
                "page": None,
            })

    return records


def chunk_document(
    record: dict[str, Any],
    strategy: str = "recursive",
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Split extracted text into chunks with stable document references."""
    if strategy == "fixed":
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    elif strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    else:
        raise ValueError("Strategy must be 'fixed' or 'recursive'.")

    chunks = splitter.split_text(record["text"])
    output = []

    for index, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        identity = (
            f"{document_id}|{record['source']}|{record['page']}|"
            f"{strategy}|{index}|{chunk}"
            if document_id
            else
            f"{record['source']}|{record['page']}|"
            f"{strategy}|{index}|{chunk}"
        )

        stable_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        output.append({
            "id": stable_id,
            "document_id": document_id,
            "text": chunk,
            "source": record["source"],
            "page": record["page"],
            "chunk_index": index,
            "chunking_strategy": strategy,
            "character_count": len(chunk),
        })

    return output


def upload_original_document(
    path: str | Path,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Upload an original file and register it in Supabase."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    document_id = document_id or str(uuid.uuid4())
    storage_path = f"{document_id}/{path.name}"

    client = get_supabase_client()
    content_type = (
        mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )

    # Upload first. The bucket must already exist and be private.
    with path.open("rb") as file_handle:
        client.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path,
            file=file_handle,
            file_options={
                "content-type": content_type,
                "upsert": "false",
            },
        )

    try:
        response = client.table("documents").insert({
            "id": document_id,
            "filename": path.name,
            "storage_path": storage_path,
            "file_type": path.suffix.lower(),
            "file_size": path.stat().st_size,
            "status": "uploaded",
        }).execute()

        return {
            "document_id": document_id,
            "storage_path": storage_path,
            "document": response.data[0] if response.data else None,
        }

    except Exception:
        # Best-effort cleanup if database registration fails.
        try:
            client.storage.from_(SUPABASE_BUCKET).remove(
                [storage_path]
            )
        except Exception:
            pass
        raise


def ingest_file(
    path: str | Path,
    strategy: str = "recursive",
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract, chunk and save processed data for one file."""
    path = Path(path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if not path.is_file():
        raise FileNotFoundError(path)

    # Keep the local copy for development and CLI workflows.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / path.name
    if path.resolve() != raw_path.resolve():
        raw_path.write_bytes(path.read_bytes())

    records = load_document(path)
    chunks = []

    for record in records:
        chunks.extend(
            chunk_document(
                record,
                strategy=strategy,
                document_id=document_id,
            )
        )

    output_path = PROCESSED_DIR / f"{path.stem}.json"
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return chunks


if __name__ == "__main__":
    files = [
        path for path in RAW_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"Add supported documents to: {RAW_DIR}")
    else:
        for path in files:
            chunks = ingest_file(path)
            print(f"{path.name}: {len(chunks)} chunks created")
