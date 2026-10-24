
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import BASE_DIR, EMBEDDING_MODEL


# ---------------- Configuration ----------------

CHROMA_DIR = BASE_DIR / "data" / "indexes" / "chroma"
BM25_PATH = BASE_DIR / "data" / "indexes" / "bm25_index.json"

COLLECTION_NAME = "internal_documents"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    """Use the same tokenization approach as the indexing phase."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------- Load persistent indexes ----------------

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_collection():
    if not CHROMA_DIR.exists():
        raise RuntimeError("ChromaDB index not found. Build indexes first.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError("ChromaDB collection is empty.")

    return collection


@lru_cache(maxsize=1)
def get_bm25_data():
    if not BM25_PATH.exists():
        raise RuntimeError("BM25 index not found. Build indexes first.")

    payload = json.loads(BM25_PATH.read_text(encoding="utf-8"))

    chunks = payload["chunks"]
    corpus = payload["tokenized_corpus"]

    if len(chunks) != len(corpus):
        raise RuntimeError("BM25 corpus and chunk counts do not match.")

    bm25 = BM25Okapi(corpus)

    return bm25, chunks


@lru_cache(maxsize=1)
def get_reranker():
    # Run reranking on CPU to avoid competing with embeddings for VRAM.
    return CrossEncoder(RERANKER_MODEL, device="cpu")


# ---------------- Dense retrieval ----------------

def dense_search(question: str, top_k: int = 10):
    collection = get_chroma_collection()
    model = get_embedding_model()

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        hits.append({
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
            "dense_rank": rank,
            "dense_distance": float(distance),
            "dense_similarity": 1.0 - float(distance),
        })

    return hits


# ---------------- Sparse retrieval ----------------

def sparse_search(question: str, top_k: int = 10):
    bm25, chunks = get_bm25_data()

    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    hits = []

    for index in ranked_indices:
        score = float(scores[index])

        # Do not treat documents with no keyword matches as useful hits.
        chunk_tokens = set(tokenize(chunks[index]["text"]))

        if not set(query_tokens).intersection(chunk_tokens):
            continue

        chunk = chunks[index]

        hits.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": {
                key: value
                for key, value in chunk.items()
                if key not in {"id", "text"}
                and value is not None
            },
            "bm25_rank": len(hits) + 1,
            "bm25_score": score,
        })

        if len(hits) >= top_k:
            break

    return hits


# ---------------- Reciprocal Rank Fusion ----------------

def reciprocal_rank_fusion(
    dense_hits,
    sparse_hits,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
):
    """Fuse ranked lists without comparing incompatible raw scores."""
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights cannot be negative.")

    if dense_weight + sparse_weight == 0:
        raise ValueError("At least one RRF weight must be positive.")

    fused = {}

    for channel, hits, weight in [
        ("dense", dense_hits, dense_weight),
        ("sparse", sparse_hits, sparse_weight),
    ]:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **hit,
                    "dense_rank": None,
                    "dense_distance": None,
                    "dense_similarity": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            item = fused[chunk_id]

            # Keep the same chunk text and metadata whichever channel found it.
            item["text"] = hit["text"]
            item["metadata"] = hit["metadata"]

            if channel == "dense":
                item["dense_rank"] = rank
                item["dense_distance"] = hit["dense_distance"]
                item["dense_similarity"] = hit["dense_similarity"]
            else:
                item["bm25_rank"] = rank
                item["bm25_score"] = hit["bm25_score"]

            item["rrf_score"] += weight / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


# ---------------- Cross-encoder reranking ----------------

def rerank(question: str, candidates: list[dict[str, Any]]):
    if not candidates:
        return []

    model = get_reranker()

    pairs = [
        (question, candidate["text"])
        for candidate in candidates
    ]

    scores = model.predict(pairs, show_progress_bar=False)

    ranked = []

    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)

    return sorted(
        ranked,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )


# ---------------- Complete retrieval pipeline ----------------

def retrieve(
    question: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    candidate_k: int = 20,
    final_k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if min(dense_k, sparse_k, candidate_k, final_k) < 1:
        raise ValueError("Retrieval limits must be positive.")

    dense_hits = dense_search(question, dense_k)
    sparse_hits = sparse_search(question, sparse_k)

    fused_hits = reciprocal_rank_fusion(
        dense_hits,
        sparse_hits,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    candidates = fused_hits[:candidate_k]
    reranked_hits = rerank(question, candidates)

    return {
        "question": question,
        "dense_results": dense_hits,
        "sparse_results": sparse_hits,
        "fused_results": fused_hits,
        "results": reranked_hits[:final_k],
    }
    
def clear_retrieval_caches() -> None:
    """Reload persistent indexes after documents are reindexed."""
    get_chroma_collection.cache_clear()
    get_bm25_data.cache_clear()


if __name__ == "__main__":
    question = input("Ask a question about your documents: ").strip()

    try:
        output = retrieve(question)

        print("\n--- Dense Search ---")
        for hit in output["dense_results"]:
            print(
                f'{hit["dense_rank"]}. {hit["metadata"].get("source")} '
                f'| similarity={hit["dense_similarity"]:.4f}'
            )
            print(hit["text"][:250], "\n")

        print("\n--- BM25 Search ---")
        for hit in output["sparse_results"]:
            print(
                f'{hit["bm25_rank"]}. {hit["metadata"].get("source")} '
                f'| score={hit["bm25_score"]:.4f}'
            )
            print(hit["text"][:250], "\n")

        print("\n--- Final Reranked Results ---")
        for rank, hit in enumerate(output["results"], start=1):
            print(
                f'\n{rank}. {hit["metadata"].get("source")} '
                f'| RRF={hit["rrf_score"]:.5f} '
                f'| reranker={hit["rerank_score"]:.4f}'
            )
            print(hit["text"])

    except Exception as exc:
        print(f"Retrieval failed: {exc}")
        raise
