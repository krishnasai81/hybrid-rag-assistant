
from typing import Any

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.supabase_client import get_supabase_client

EMBEDDING_BATCH_SIZE = 32
DB_BATCH_SIZE = 100


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep source information needed for citations."""
    return {
        "source": str(chunk["source"]),
        "page": chunk.get("page"),
        "chunk_index": int(chunk["chunk_index"]),
        "chunking_strategy": str(
            chunk.get("chunking_strategy", "recursive")
        ),
        "character_count": int(
            chunk.get("character_count", len(chunk["text"]))
        ),
    }


def index_document_chunks(
    document_id: str,
    chunks: list[dict[str, Any]],
) -> int:
    """Embed one document and persist its chunks in Supabase."""
    if not chunks:
        return 0

    client = get_supabase_client()
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        if chunk.get("document_id") != document_id:
            raise ValueError(
                "Chunk document_id does not match the uploaded document."
            )

        rows.append({
            "id": chunk["id"],
            "document_id": document_id,
            "content": chunk["text"],
            "embedding": "[" + ",".join(
                str(float(value)) for value in embedding
            ) + "]",
            "chunk_index": int(chunk["chunk_index"]),
            "page": chunk.get("page"),
            "metadata": build_metadata(chunk),
        })

    for start in range(0, len(rows), DB_BATCH_SIZE):
        batch = rows[start:start + DB_BATCH_SIZE]
        client.table("document_chunks").upsert(
            batch,
            on_conflict="id",
        ).execute()

    client.table("documents").update({
        "status": "processed",
    }).eq("id", document_id).execute()

    return len(rows)
