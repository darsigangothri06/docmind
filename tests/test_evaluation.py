import json
import tempfile
from pathlib import Path
from src.evaluation.dataset import EvalDataset


def test_eval_dataset_loading():
    data = [
        {"question": "What is X?", "ground_truth": "X is Y."},
        {"question": "How does Z work?", "ground_truth": "Z works via A."},
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        f.flush()
        dataset = EvalDataset().load(f.name)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "What is X?"


def test_evaluator_interface():
    from src.evaluation.metrics import RAGEvaluator
    assert hasattr(RAGEvaluator, "evaluate")
    assert hasattr(RAGEvaluator, "faithfulness")
    assert hasattr(RAGEvaluator, "answer_relevance")
    assert hasattr(RAGEvaluator, "context_precision")
    assert hasattr(RAGEvaluator, "context_recall")
