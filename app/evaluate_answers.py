
import json
from pathlib import Path
from typing import Any

from app.generation import ask_documents


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "answer_results.json"


ABSTENTION_PHRASES = (
    "does not answer the question",
    "does not specify",
    "doesn't specify",
    "not specified",
    "insufficient evidence",
    "cannot determine",
    "can't determine",
    "could not find information",
    "couldn't find information",
    "not enough information",
    "unable to determine",
    "does not contain",
)



def is_abstention(answer: str) -> bool:
    """Recognize common ways the assistant says evidence is missing."""
    text = " ".join(answer.casefold().split())

    patterns = (
        "does not answer the question",
        "doesn't answer the question",
        "does not specify",
        "doesn't specify",
        "does not mention",
        "doesn't mention",
        "not mentioned",
        "not specified",
        "does not contain information",
        "doesn't contain information",
        "no information about",
        "insufficient evidence",
        "insufficient information",
        "cannot determine",
        "can't determine",
        "could not determine",
        "couldn't determine",
        "could not find information",
        "couldn't find information",
        "not enough information",
        "unable to determine",
        "the evidence is insufficient",
        "does not contain any information",
        "doesn't contain any information",
    )

    return any(pattern in text for pattern in patterns)


def get_verification(result: dict[str, Any]) -> dict[str, Any]:
    verification = result.get("verification", {})
    return verification if isinstance(verification, dict) else {}


def evaluate_answers() -> None:
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    report = []
    unanswerable_total = 0
    unanswerable_abstentions = 0
    answerable_total = 0
    answerable_responses = 0
    citation_coverages = []
    invalid_citation_total = 0
    unsupported_claim_total = 0

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        answerable = item["answerable"]

        print(f"\n{'=' * 65}")
        print(f"{question_id}: {question}")

        try:
            result = ask_documents(question)

            if not isinstance(result, dict):
                raise TypeError(
                    "ask_documents() must return a dictionary."
                )

            answer = str(result.get("answer", ""))
            sources = result.get("sources", [])
            verification = get_verification(result)

            abstained = is_abstention(answer)

            if answerable:
                answerable_total += 1
                if answer.strip() and not abstained:
                    answerable_responses += 1
            else:
                unanswerable_total += 1
                if abstained:
                    unanswerable_abstentions += 1

            invalid_citations = verification.get(
                "invalid_citations", []
            )
            invalid_citation_total += len(invalid_citations)

            coverage = verification.get("citation_coverage")
            if isinstance(coverage, (int, float)):
                citation_coverages.append(float(coverage))

            checks = verification.get("checks", [])
            for check in checks:
                if check.get("verdict") == "unsupported":
                    unsupported_claim_total += 1

            entry = {
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "answerable": answerable,
                "generated_answer": answer,
                "abstained": abstained,
                "sources": sources,
                "verification": verification,
            }
            report.append(entry)

            print(f"\nExpected answer: {expected_answer}")
            print(f"Generated answer: {answer}")
            print(f"Abstained: {abstained}")
            print(f"Sources: {json.dumps(sources, ensure_ascii=False)}")
            print(
                "Citation coverage:",
                verification.get("citation_coverage", "unavailable"),
            )
            print("Invalid citations:", invalid_citations)

        except Exception as exc:
            print(f"ERROR: {exc}")
            report.append({
                "id": question_id,
                "question": question,
                "answerable": answerable,
                "error": str(exc),
            })

    summary = {
        "total_questions": len(dataset),
        "answerable_questions": answerable_total,
        "answerable_response_rate": (
            answerable_responses / answerable_total
            if answerable_total else 0.0
        ),
        "unanswerable_questions": unanswerable_total,
        "abstention_rate_on_unanswerable": (
            unanswerable_abstentions / unanswerable_total
            if unanswerable_total else 0.0
        ),
        "average_citation_coverage": (
            sum(citation_coverages) / len(citation_coverages)
            if citation_coverages else None
        ),
        "invalid_citation_count": invalid_citation_total,
        "unsupported_claim_count": unsupported_claim_total,
        "note": (
            "Response rate is not answer correctness. "
            "Review generated answers against expected answers."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "results": report},
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'=' * 65}")
    print("ANSWER EVALUATION SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_answers()
