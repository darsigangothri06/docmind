from __future__ import annotations

from .metrics import RAGEvaluator
from .dataset import EvalDataset
from ..generation.chain import RAGChain


class EvaluationOrchestrator:
    """Runs evaluation across a test dataset and aggregates results."""

    def __init__(self, rag_chain: RAGChain, llm):
        self.rag_chain = rag_chain
        self.evaluator = RAGEvaluator(llm)

    def run(self, dataset_path: str) -> dict:
        dataset = EvalDataset().load(dataset_path)
        results = []

        for item in dataset:
            question = item["question"]
            ground_truth = item.get("ground_truth")

            response = self.rag_chain.query(question)
            contexts = [s["content"] for s in response.sources]

            scores = self.evaluator.evaluate(
                question=question,
                answer=response.answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

            results.append({
                "question": question,
                "answer": response.answer,
                "scores": scores,
            })

        avg_scores = self._aggregate(results)
        return {"results": results, "average_scores": avg_scores}

    def _aggregate(self, results: list[dict]) -> dict:
        if not results:
            return {}
        all_keys = set()
        for r in results:
            all_keys.update(r["scores"].keys())

        averages = {}
        for key in all_keys:
            values = [r["scores"][key] for r in results if key in r["scores"]]
            averages[key] = round(sum(values) / len(values), 3) if values else 0.0
        return averages
