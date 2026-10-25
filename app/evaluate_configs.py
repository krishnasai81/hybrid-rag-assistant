
import json
import re
from pathlib import Path

from app.retrieval import retrieve

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "retrieval_comparison.json"

K_VALUES = (1, 3, 5)


def normalize(text: str) -> str:
    """Normalize case and whitespace without changing meaning."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_relevant(chunk: dict, anchor: str) -> bool:
    text = chunk.get("text", "")
    return bool(anchor) and normalize(anchor) in normalize(text)


def calculate_metrics(results: list, anchor: str) -> dict:
    relevant_ranks = [
        rank
        for rank, chunk in enumerate(results, start=1)
        if is_relevant(chunk, anchor)
    ]

    first_rank = min(relevant_ranks) if relevant_ranks else None

    metrics = {
        "first_relevant_rank": first_rank,
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "relevant_ranks": relevant_ranks,
    }

    for k in K_VALUES:
        relevant_count = sum(rank <= k for rank in relevant_ranks)
        metrics[f"precision_at_{k}"] = relevant_count / k
        metrics[f"hit_rate_at_{k}"] = int(relevant_count > 0)

    return metrics


def evaluate() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    answerable = [item for item in dataset if item.get("answerable", False)]

    configurations = {
        "dense_only": [],
        "bm25_only": [],
        "hybrid_rrf": [],
        "hybrid_rrf_cross_encoder": [],
    }

    question_results = []

    for index, item in enumerate(answerable, start=1):
        question = item["question"]
        anchor = item["relevant_text_contains"]

        print(f"\n[{index}/{len(answerable)}] {item['id']}: {question}")

        output = retrieve(
            question,
            dense_k=10,
            sparse_k=10,
            candidate_k=20,
            final_k=5,
        )

        variants = {
            "dense_only": output["dense_results"][:5],
            "bm25_only": output["sparse_results"][:5],
            "hybrid_rrf": output["fused_results"][:5],
            "hybrid_rrf_cross_encoder": output["results"][:5],
        }

        row = {"id": item["id"], "question": question}

        for name, results in variants.items():
            metrics = calculate_metrics(results, anchor)
            configurations[name].append(metrics)
            row[name] = metrics

            print(
                f"{name}: first relevant rank="
                f"{metrics['first_relevant_rank']}, "
                f"MRR={metrics['reciprocal_rank']:.3f}, "
                f"P@5={metrics['precision_at_5']:.3f}"
            )

        question_results.append(row)

    summary = {}

    for name, rows in configurations.items():
        count = len(rows)
        summary[name] = {
            "questions_evaluated": count,
            "mrr": (
                sum(row["reciprocal_rank"] for row in rows) / count
                if count else 0.0
            ),
        }

        for k in K_VALUES:
            summary[name][f"precision_at_{k}"] = (
                sum(row[f"precision_at_{k}"] for row in rows) / count
                if count else 0.0
            )
            summary[name][f"hit_rate_at_{k}"] = (
                sum(row[f"hit_rate_at_{k}"] for row in rows) / count
                if count else 0.0
            )

    report = {
        "evaluation_type": "Retrieval configuration comparison",
        "relevance_method": "Case-insensitive exact evidence phrase matching",
        "note": (
            "Results depend on the quality of manually verified evidence "
            "anchors. Hit Rate@K is not full corpus-level Recall@K."
        ),
        "summary": summary,
        "question_results": question_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== CONFIGURATION COMPARISON ==========")

    for name, metrics in summary.items():
        print(
            f"{name:28} "
            f"MRR={metrics['mrr']:.3f} | "
            f"P@1={metrics['precision_at_1']:.3f} | "
            f"P@3={metrics['precision_at_3']:.3f} | "
            f"P@5={metrics['precision_at_5']:.3f} | "
            f"Hit@5={metrics['hit_rate_at_5']:.3f}"
        )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
