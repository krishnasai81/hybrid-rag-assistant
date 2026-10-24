
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import PROCESSED_DIR, EMBEDDING_MODEL, BASE_DIR


INDEX_DIR = BASE_DIR / "data" / "indexes"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25_index.json"

COLLECTION_NAME = "internal_documents"
BATCH_SIZE = 128
EMBEDDING_BATCH_SIZE = 32

INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    """Tokenize text for case-insensitive BM25 retrieval."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks() -> list[dict[str, Any]]:
    """Load and validate all processed chunks."""
    chunks = []
    seen_ids = set()

    for path in sorted(PROCESSED_DIR.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list of chunks in {path.name}"
            )

        for chunk in records:
            required = {"id", "text", "source", "chunk_index"}

            if not isinstance(chunk, dict) or not required.issubset(chunk):
                raise ValueError(
                    f"Invalid chunk structure in {path.name}"
                )

            if not isinstance(chunk["id"], str) or not chunk["id"]:
                raise ValueError(
                    f"Invalid chunk ID in {path.name}"
                )

            if not isinstance(chunk["text"], str):
                raise ValueError(
                    f"Invalid chunk text in {path.name}"
                )

            if not chunk["text"].strip():
                continue

            if chunk["id"] in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID found: {chunk['id']}"
                )

            seen_ids.add(chunk["id"])
            chunks.append(chunk)

    return chunks


def get_collection():
    """Open the existing collection or create it if necessary."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        # Do not hide unrelated ChromaDB errors.
        if "does not exist" not in str(exc).lower():
            raise

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return client, collection


def get_all_collection_ids(collection) -> set[str]:
    """Fetch IDs in pages to avoid one huge collection.get() call."""
    existing_ids = set()
    offset = 0

    while True:
        result = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=[],
        )

        batch_ids = result.get("ids", [])

        if not batch_ids:
            break

        existing_ids.update(batch_ids)
        offset += len(batch_ids)

    return existing_ids


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert chunk metadata to ChromaDB-compatible scalar values."""
    metadata = {
        "source": str(chunk["source"]),
        "chunk_index": int(chunk["chunk_index"]),
        "chunking_strategy": str(
            chunk.get("chunking_strategy", "unknown")
        ),
        "character_count": int(
            chunk.get("character_count", len(chunk["text"]))
        ),
    }

    if chunk.get("page") is not None:
        metadata["page"] = int(chunk["page"])

    if chunk.get("section_heading"):
        metadata["section_heading"] = str(
            chunk["section_heading"]
        )

    return metadata


def write_bm25_index(chunks: list[dict[str, Any]]) -> None:
    """Rebuild the BM25 corpus from the current processed documents."""
    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    payload = {
        "version": 1,
        "tokenized_corpus": tokenized_corpus,
        "chunks": chunks,
    }

    temporary_path = BM25_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(BM25_PATH)


def build_indexes() -> None:
    """
    Synchronize ChromaDB with processed chunks.

    New chunks receive embeddings only once.
    Removed or changed chunks are reconciled by ID.
    BM25 data is regenerated from the current processed corpus.
    """
    chunks = load_chunks()

    client, collection = get_collection()

    desired_by_id = {
        chunk["id"]: chunk
        for chunk in chunks
    }
    desired_ids = set(desired_by_id)

    existing_ids = get_all_collection_ids(collection)

    ids_to_add = desired_ids - existing_ids
    ids_to_remove = existing_ids - desired_ids

    print(f"Processed chunks: {len(chunks)}")
    print(f"Existing ChromaDB chunks: {len(existing_ids)}")
    print(f"New chunks to embed: {len(ids_to_add)}")
    print(f"Stale chunks to remove: {len(ids_to_remove)}")

    # Embed and insert only chunks that are not already indexed.
    if ids_to_add:
        model = SentenceTransformer(EMBEDDING_MODEL)

        new_chunks = [
            chunk
            for chunk in chunks
            if chunk["id"] in ids_to_add
        ]

        for start in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[start:start + BATCH_SIZE]
            texts = [chunk["text"] for chunk in batch]

            embeddings = model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=texts,
                metadatas=[
                    build_metadata(chunk)
                    for chunk in batch
                ],
                embeddings=embeddings.tolist(),
            )

    # Remove chunks whose IDs no longer exist in processed data.
    if ids_to_remove:
        stale_ids = list(ids_to_remove)

        for start in range(0, len(stale_ids), BATCH_SIZE):
            collection.delete(
                ids=stale_ids[start:start + BATCH_SIZE]
            )

    # Rebuild keyword-search data from the current corpus.
    # This does not regenerate embeddings.
    write_bm25_index(chunks)

    stored_count = collection.count()

    if stored_count != len(chunks):
        raise RuntimeError(
            f"Index count mismatch: expected {len(chunks)}, "
            f"found {stored_count}"
        )

    print("\nIndex synchronization completed.")
    print(f"ChromaDB chunks: {stored_count}")
    print(f"BM25 chunks: {len(chunks)}")
    print(f"ChromaDB path: {CHROMA_DIR}")
    print(f"BM25 index: {BM25_PATH}")


if __name__ == "__main__":
    build_indexes()
            