
import json
from pathlib import Path

from app.generation import ask_documents

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "answer_evaluation.json"


def evaluate_answers():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    report = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        print(f"\n[{index}/{len(dataset)}] {item['id']}: {question}")

        try:
            result = ask_documents(question)

            row = {
                "id": item["id"],
                "question": question,
                "answerable": item["answerable"],
                "reference_answer": item.get("reference_answer", ""),
                "generated_answer": result.get("answer", ""),
                "retrieved_contexts": [
                    source.get("text", "")
                    for source in result.get("sources", [])
                ],
                "sources": result.get("sources", []),
                "verification": result.get("verification", {}),
                "error": None,
            }

            print("Answer:", row["generated_answer"])
            print(
                "Citation coverage:",
                row["verification"].get("citation_coverage", 0.0),
            )

        except Exception as exc:
            row = {
                "id": item["id"],
                "question": question,
                "answerable": item["answerable"],
                "reference_answer": item.get("reference_answer", ""),
                "generated_answer": "",
                "retrieved_contexts": [],
                "sources": [],
                "verification": {},
                "error": str(exc),
            }
            print("ERROR:", exc)

        report.append(row)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nEvaluation report saved to: {REPORT_PATH}")
    print(f"Questions processed: {len(report)}")
    print(f"Errors: {sum(row['error'] is not None for row in report)}")


if __name__ == "__main__":
    evaluate_answers()
