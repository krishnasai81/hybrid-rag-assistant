
import asyncio
import json
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecisionWithReference,
    ContextRecall,
    AnswerRelevancy,
    FactualCorrectness,
    NoiseSensitivity,
    ContextEntityRecall,
    ResponseGroundedness,
)

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.generation import ask_documents


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "golden_dataset.json"
REPORT_PATH = BASE_DIR / "data" / "evaluation" / "ragas_results.json"


async def main():
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your project .env file."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    evaluator_llm = llm_factory(
        GROQ_MODEL,
        client=client,
    )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "context_precision": ContextPrecisionWithReference(
            llm=evaluator_llm
        ),
        "context_recall": ContextRecall(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=embeddings,
        ),
        "factual_correctness": FactualCorrectness(
            llm=evaluator_llm
        ),
        "noise_sensitivity": NoiseSensitivity(
            llm=evaluator_llm
        ),
        "context_entity_recall": ContextEntityRecall(
            llm=evaluator_llm
        ),
        "response_groundedness": ResponseGroundedness(
            llm=evaluator_llm
        ),
    }

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    results = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        reference = item["expected_answer"]
        answerable = item.get("answerable", True)

        print(f"\n[{index}/{len(dataset)}] {question}")

        try:
            rag_result = ask_documents(question)

            answer = rag_result.get("answer", "")
            contexts = [
                source["text"]
                for source in rag_result.get("sources", [])
                if source.get("text", "").strip()
            ]

            row = {
                "id": item.get("id", str(index)),
                "question": question,
                "answerable": answerable,
                "answer": answer,
                "retrieved_context_count": len(contexts),
                "metrics": {},
                "errors": {},
            }

            if not contexts:
                row["errors"]["retrieval"] = (
                    "No passages were retrieved for this question."
                )

            for name, metric in metrics.items():
                # Reference-dependent metrics are not meaningful for
                # questions whose expected behavior is abstention.
                if not answerable and name in {
                    "context_precision",
                    "context_recall",
                    "factual_correctness",
                    "noise_sensitivity",
                    "context_entity_recall",
                }:
                    row["metrics"][name] = None
                    continue

                # These metrics need retrieved evidence.
                if not contexts and name in {
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                    "noise_sensitivity",
                    "context_entity_recall",
                    "response_groundedness",
                }:
                    row["metrics"][name] = None
                    continue

                try:
                    if name == "faithfulness":
                        arguments = {
                            "user_input": question,
                            "response": answer,
                            "retrieved_contexts": contexts,
                        }

                    elif name == "context_precision":
                        arguments = {
                            "user_input": question,
                            "reference": reference,
                            "retrieved_contexts": contexts,
                        }

                    elif name == "context_recall":
                        arguments = {
                            "user_input": question,
                            "reference": reference,
                            "retrieved_contexts": contexts,
                        }

                    elif name == "answer_relevancy":
                        arguments = {
                            "user_input": question,
                            "response": answer,
                        }

                    elif name == "factual_correctness":
                        arguments = {
                            "response": answer,
                            "reference": reference,
                        }

                    elif name == "noise_sensitivity":
                        arguments = {
                            "user_input": question,
                            "response": answer,
                            "reference": reference,
                            "retrieved_contexts": contexts,
                        }

                    elif name == "context_entity_recall":
                        arguments = {
                            "reference": reference,
                            "retrieved_contexts": contexts,
                        }

                    elif name == "response_groundedness":
                        arguments = {
                            "response": answer,
                            "retrieved_contexts": contexts,
                        }

                    else:
                        row["metrics"][name] = None
                        continue

                    score = await metric.ascore(**arguments)
                    row["metrics"][name] = float(score.value)

                    print(f"  {name}: {score.value}")

                except Exception as exc:
                    row["metrics"][name] = None
                    row["errors"][name] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"  {name} failed: {exc}")

            results.append(row)

        except Exception as exc:
            print(f"  Question failed: {exc}")

            results.append({
                "id": item.get("id", str(index)),
                "question": question,
                "answerable": answerable,
                "error": f"{type(exc).__name__}: {exc}",
            })

    metric_names = list(metrics.keys())
    summary = {}

    for name in metric_names:
        values = [
            row.get("metrics", {}).get(name)
            for row in results
            if isinstance(
                row.get("metrics", {}).get(name),
                (int, float),
            )
        ]

        summary[name] = {
            "mean_score": round(mean(values), 4) if values else None,
            "scored_questions": len(values),
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "ragas_version": "0.4.3",
        "total_questions": len(dataset),
        "summary": summary,
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n--- RAGAS SUMMARY ---")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
