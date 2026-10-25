
import re
from functools import lru_cache
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config import EMBEDDING_MODEL
from app.supabase_client import get_supabase_client

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DB_PAGE_SIZE = 500


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def load_all_chunks() -> list[dict[str, Any]]:
    """Load the persisted chunk corpus in pages for BM25."""
    client = get_supabase_client()
    chunks = []
    start = 0

    while True:
        response = (
            client.table("document_chunks")
            .select(
                "id,document_id,content,chunk_index,page,metadata"
            )
            .order("id")
            .range(start, start + DB_PAGE_SIZE - 1)
            .execute()
        )

        batch = response.data or []
        if not batch:
            break

        for row in batch:
            metadata = row.get("metadata") or {}
            metadata = dict(metadata)
            metadata.setdefault("chunk_index", row["chunk_index"])
            metadata.setdefault("page", row.get("page"))

            chunks.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "text": row["content"],
                "metadata": metadata,
            })

        if len(batch) < DB_PAGE_SIZE:
            break

        start += len(batch)

    return chunks


@lru_cache(maxsize=1)
def get_bm25_data():
    chunks = load_all_chunks()
    if not chunks:
        raise RuntimeError(
            "No document chunks found in Supabase. Upload a document first."
        )

    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    return BM25Okapi(corpus), chunks


def dense_search(question: str, top_k: int = 10):
    client = get_supabase_client()
    model = get_embedding_model()

    embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    response = client.rpc(
        "match_document_chunks",
        {
            "query_embedding": "[" + ",".join(
                str(float(value)) for value in embedding
            ) + "]",
            "match_count": top_k,
        },
    ).execute()

    hits = []
    for rank, row in enumerate(response.data or [], start=1):
        metadata = dict(row.get("metadata") or {})
        metadata.setdefault("page", row.get("page"))
        metadata.setdefault("chunk_index", row.get("chunk_index"))

        hits.append({
            "id": row["id"],
            "text": row["content"],
            "metadata": metadata,
            "dense_rank": rank,
            "dense_similarity": float(row["similarity"]),
            "dense_distance": 1.0 - float(row["similarity"]),
        })

    return hits


def sparse_search(question: str, top_k: int = 10):
    bm25, chunks = get_bm25_data()
    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked_indices = np.argsort(scores)[::-1]

    hits = []
    for index in ranked_indices:
        chunk = chunks[int(index)]
        chunk_terms = set(tokenize(chunk["text"]))

        if not query_terms.intersection(chunk_terms):
            continue

        hits.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "bm25_rank": len(hits) + 1,
            "bm25_score": float(scores[index]),
        })

        if len(hits) >= top_k:
            break

    return hits


def reciprocal_rank_fusion(
    dense_hits,
    sparse_hits,
    dense_weight=0.7,
    sparse_weight=0.3,
    rrf_k=60,
):
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights cannot be negative.")
    if dense_weight + sparse_weight == 0:
        raise ValueError("At least one RRF weight must be positive.")

    fused = {}

    for channel, hits, weight in (
        ("dense", dense_hits, dense_weight),
        ("sparse", sparse_hits, sparse_weight),
    ):
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
            item["text"] = hit["text"]
            item["metadata"] = hit["metadata"]

            if channel == "dense":
                item["dense_rank"] = rank
                item["dense_distance"] = hit.get("dense_distance")
                item["dense_similarity"] = hit.get("dense_similarity")
            else:
                item["bm25_rank"] = rank
                item["bm25_score"] = hit.get("bm25_score")

            item["rrf_score"] += weight / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def rerank(question: str, candidates):
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(question, item["text"]) for item in candidates]
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


def retrieve(
    question: str,
    dense_k=10,
    sparse_k=10,
    candidate_k=20,
    final_k=5,
    dense_weight=0.7,
    sparse_weight=0.3,
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

    reranked_hits = rerank(
        question,
        fused_hits[:candidate_k],
    )

    return {
        "question": question,
        "dense_results": dense_hits,
        "sparse_results": sparse_hits,
        "fused_results": fused_hits,
        "results": reranked_hits[:final_k],
    }


def clear_retrieval_caches():
    get_bm25_data.cache_clear()
    get_embedding_model.cache_clear()
    get_reranker.cache_clear()
