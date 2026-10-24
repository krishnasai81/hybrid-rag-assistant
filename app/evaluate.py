
import json
from pathlib import Path
from typing import Any

from app.retrieval import retrieve


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "retrieval_results.json"

K_VALUES = (1, 3, 5)


def extract_text(result: dict[str, Any]) -> str:
    """Extract the document text from a retrieved chunk."""
    for key in ("text", "document", "content", "chunk_text"):
        value = result.get(key)
        if isinstance(value, str):
            return value

    metadata = result.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("text", "document", "content", "chunk_text"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value

    return ""


def is_relevant(result: dict[str, Any], anchor: str) -> bool:
    """Check whether a retrieved chunk contains the expected evidence."""
    if not anchor.strip():
        return False

    return anchor.casefold() in extract_text(result).casefold()


def evaluate() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    report = []
    reciprocal_ranks = []
    hits_at_k = {k: [] for k in K_VALUES}

    answerable_questions = [
        item for item in dataset if item["answerable"]
    ]

    if not answerable_questions:
        raise ValueError("The dataset contains no answerable questions.")

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        anchor = item["relevant_text_contains"]

        print(f"\nEvaluating {question_id}: {question}")

        # Evaluate retrieval only for questions with known evidence.
        if not item["answerable"]:
            report.append({
                "id": question_id,
                "question": question,
                "answerable": False,
                "retrieval_metrics_applicable": False,
                "note": "Evaluate abstention separately in Phase 4.3."
            })
            print("Skipped: unanswerable question.")
            continue

        output = retrieve(question, final_k=max(K_VALUES))
        print("\nDEBUG: Output type:", type(output))
        print("DEBUG: Output keys:", output.keys() if isinstance(output, dict) else "Not a dictionary")
        print("DEBUG: First result:", output.get("final", [])[:1] if isinstance(output, dict) else output)
        retrieved = output.get("results", [])

        if not isinstance(retrieved, list):
            raise TypeError(
                "Expected retrieve() to return a list in output['final']."
            )

        relevant_ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if is_relevant(chunk, anchor)
        ]

        first_rank = min(relevant_ranks) if relevant_ranks else None
        reciprocal_rank = 1 / first_rank if first_rank else 0.0

        reciprocal_ranks.append(reciprocal_rank)

        question_metrics = {}
        for k in K_VALUES:
            hit = any(rank <= k for rank in relevant_ranks)
            hits_at_k[k].append(int(hit))
            question_metrics[f"recall_at_{k}"] = int(hit)

        result = {
            "id": question_id,
            "question": question,
            "answerable": True,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": reciprocal_rank,
            **question_metrics,
            "retrieved_chunks": len(retrieved),
        }

        report.append(result)

        print(f"First relevant rank: {first_rank}")
        print(f"MRR contribution: {reciprocal_rank:.3f}")
        print(f"Recall@1: {question_metrics['recall_at_1']}")
        print(f"Recall@3: {question_metrics['recall_at_3']}")
        print(f"Recall@5: {question_metrics['recall_at_5']}")

    summary = {
        "answerable_questions_evaluated": len(answerable_questions),
        "recall_at_1": sum(hits_at_k[1]) / len(hits_at_k[1]),
        "recall_at_3": sum(hits_at_k[3]) / len(hits_at_k[3]),
        "recall_at_5": sum(hits_at_k[5]) / len(hits_at_k[5]),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }

    final_report = {
        "summary": summary,
        "question_results": report,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=2, ensure_ascii=False)

    print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")
    print(f"Questions evaluated: {summary['answerable_questions_evaluated']}")
    print(f"Recall@1: {summary['recall_at_1']:.3f}")
    print(f"Recall@3: {summary['recall_at_3']:.3f}")
    print(f"Recall@5: {summary['recall_at_5']:.3f}")
    print(f"MRR:      {summary['mrr']:.3f}")
    print(f"\nDetailed report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
