
import json
import re
from pathlib import Path
from typing import Any

from app.retrieval import retrieve


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "retrieval_results.json"

K_VALUES = (1, 3, 5)

# Common words contribute little to identifying relevant evidence.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "for", "from", "how", "in", "is", "it", "must", "of",
    "on", "or", "the", "their", "this", "to", "what", "when",
    "where", "which", "who", "with",
}


def normalize_word(word: str) -> str:
    """Normalize common English word forms for simple matching."""
    word = word.casefold()

    # Handle common suffixes without requiring another dependency.
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[:-len(suffix)]
            break

    return word


def tokenize(text: str) -> list[str]:
    """Extract normalized word tokens, ignoring punctuation."""
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [normalize_word(word) for word in words]


def is_relevant(
    chunk: dict[str, Any],
    anchor: str | list[str],
    threshold: float = 0.60,
) -> bool:
    """
    Match evidence using normalized text and keyword coverage.

    An anchor may be a string or a list of alternative evidence phrases.
    """
    text = chunk.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return False

    anchors = [anchor] if isinstance(anchor, str) else anchor

    for phrase in anchors:
        if not isinstance(phrase, str) or not phrase.strip():
            continue

        # First try normalized exact phrase matching.
        normalized_text = " ".join(tokenize(text))
        normalized_anchor = " ".join(tokenize(phrase))

        if normalized_anchor and normalized_anchor in normalized_text:
            return True

        # Otherwise, measure coverage of meaningful anchor keywords.
        anchor_tokens = {
            token
            for token in tokenize(phrase)
            if token not in STOP_WORDS
        }
        chunk_tokens = set(tokenize(text))

        if len(anchor_tokens) < 2:
            continue

        coverage = len(anchor_tokens & chunk_tokens) / len(anchor_tokens)

        if coverage >= threshold:
            return True

    return False


def evaluate() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    answerable_questions = [
        item for item in dataset if item.get("answerable", False)
    ]

    if not answerable_questions:
        raise ValueError("The dataset contains no answerable questions.")

    question_results = []
    metric_values = {
        f"precision_at_{k}": [] for k in K_VALUES
    }
    metric_values.update({
        f"evidence_hit_at_{k}": [] for k in K_VALUES
    })
    metric_values.update({
        f"hit_rate_at_{k}": [] for k in K_VALUES
    })

    reciprocal_ranks = []

    for index, item in enumerate(dataset, start=1):
        question_id = item.get("id", str(index))
        question = item["question"]
        anchor = item.get("relevant_text_contains", "")

        print(f"\n[{index}/{len(dataset)}] {question_id}: {question}")

        if not item.get("answerable", False):
            question_results.append({
                "id": question_id,
                "question": question,
                "answerable": False,
                "retrieval_metrics_applicable": False,
                "note": "Skipped: no relevant evidence is specified.",
            })
            print("Skipped: unanswerable question.")
            continue

        if not anchor or (
            isinstance(anchor, str) and not anchor.strip()
        ) or (
            isinstance(anchor, list)
            and not any(isinstance(a, str) and a.strip() for a in anchor)
        ):
            question_results.append({
                "id": question_id,
                "question": question,
                "answerable": True,
                "retrieval_metrics_applicable": False,
                "note": "Skipped: no relevance anchor.",
            })
            print("Skipped: no relevance anchor.")
            continue

        output = retrieve(question, final_k=max(K_VALUES))
        retrieved = output.get("results", [])

        if not isinstance(retrieved, list):
            raise TypeError(
                "retrieve() must return a list in output['results']."
            )

        print("\nExpected evidence:", anchor)

        for rank, chunk in enumerate(retrieved, start=1):
            print(f"\n--- Retrieved chunk {rank} ---")
            print("Source:", chunk.get("metadata", {}).get("source"))
            print("Text:", chunk.get("text", "")[:500])

        relevant_ranks = [
            rank
            for rank, chunk in enumerate(retrieved, start=1)
            if is_relevant(chunk, anchor)
        ]

        first_rank = min(relevant_ranks) if relevant_ranks else None
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
        reciprocal_ranks.append(reciprocal_rank)

        question_metrics = {}

        for k in K_VALUES:
            relevant_in_top_k = sum(
                rank <= k for rank in relevant_ranks
            )

            # Precision@K uses K as the denominator, even if fewer than K
            # chunks were returned.
            precision = relevant_in_top_k / k

            # This is an evidence-hit indicator, NOT corpus-level recall.
            evidence_hit = int(relevant_in_top_k > 0)

            question_metrics[f"precision_at_{k}"] = precision
            question_metrics[f"evidence_hit_at_{k}"] = evidence_hit
            question_metrics[f"hit_rate_at_{k}"] = evidence_hit

            metric_values[f"precision_at_{k}"].append(precision)
            metric_values[f"evidence_hit_at_{k}"].append(evidence_hit)
            metric_values[f"hit_rate_at_{k}"].append(evidence_hit)

        question_results.append({
            "id": question_id,
            "question": question,
            "answerable": True,
            "retrieval_metrics_applicable": True,
            "relevant_anchor": anchor,
            "retrieved_chunks": len(retrieved),
            "relevant_ranks": relevant_ranks,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": reciprocal_rank,
            **question_metrics,
        })

        print(f"Retrieved chunks: {len(retrieved)}")
        print(f"Relevant ranks: {relevant_ranks}")
        print(f"Reciprocal rank: {reciprocal_rank:.3f}")

        for k in K_VALUES:
            print(
                f"Precision@{k}: "
                f"{question_metrics[f'precision_at_{k}']:.3f} | "
                f"Evidence-hit@{k}: "
                f"{question_metrics[f'evidence_hit_at_{k}']}"
            )

    evaluated_count = len(reciprocal_ranks)

    summary = {
        "answerable_questions_in_dataset": len(answerable_questions),
        "questions_evaluated": evaluated_count,
        "mrr": (
            sum(reciprocal_ranks) / evaluated_count
            if evaluated_count else None
        ),
    }

    for name, values in metric_values.items():
        summary[name] = sum(values) / len(values) if values else None

    final_report = {
        "evaluation_type": "Hybrid RAG retrieval evaluation",
        "k_values": list(K_VALUES),
        "relevance_method": (
            "Normalized exact matching with keyword-coverage fallback "
            "at a threshold of 0.60."
        ),
        "metric_note": (
            "Evidence-hit@K and Hit Rate@K indicate whether at least one "
            "chunk matched the relevance anchor. They are not full "
            "corpus-level Recall@K. Precision@K depends on the quality "
            "of anchor-based relevance labels."
        ),
        "summary": summary,
        "question_results": question_results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=2, ensure_ascii=False)

    print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")
    print(f"Questions evaluated: {evaluated_count}")
    print(f"MRR: {summary['mrr']:.3f}")

    for k in K_VALUES:
        print(f"Precision@{k}: {summary[f'precision_at_{k}']:.3f}")
        print(
            f"Evidence-hit@{k}: "
            f"{summary[f'evidence_hit_at_{k}']:.3f}"
        )
        print(f"Hit Rate@{k}: {summary[f'hit_rate_at_{k}']:.3f}")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
