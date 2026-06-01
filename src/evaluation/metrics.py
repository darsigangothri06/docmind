from __future__ import annotations

import json
import re


class RAGEvaluator:
    """Evaluate RAG pipeline quality across 4 dimensions using LLM-as-judge."""

    def __init__(self, llm):
        self.llm = llm

    def _parse_score(self, response_text: str) -> float:
        """Extract score from LLM response JSON."""
        try:
            match = re.search(r'"score"\s*:\s*([\d.]+)', response_text)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
        except (ValueError, AttributeError):
            pass
        return 0.0

    def faithfulness(self, answer: str, context: str) -> float:
        """Is the answer grounded in context? (0.0 - 1.0)"""
        prompt = f"""Extract factual claims from this answer, then check if each is supported by the context.

Answer: {answer}
Context: {context}

Output ONLY valid JSON (no markdown): {{"claims": [{{"claim": "...", "supported": true/false}}], "score": 0.0-1.0}}
The score should be the fraction of claims that are supported."""
        result = self.llm.invoke(prompt)
        return self._parse_score(result.content)

    def answer_relevance(self, answer: str, question: str) -> float:
        """Does the answer address the question? (0.0 - 1.0)"""
        prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0.

Question: {question}
Answer: {answer}

Output ONLY valid JSON (no markdown): {{"reasoning": "brief explanation", "score": 0.0-1.0}}"""
        result = self.llm.invoke(prompt)
        return self._parse_score(result.content)

    def context_precision(self, contexts: list[str], question: str) -> float:
        """Are the retrieved chunks relevant? (0.0 - 1.0)"""
        context_list = "\n---\n".join(f"Chunk {i+1}: {c[:300]}" for i, c in enumerate(contexts))
        prompt = f"""For each context chunk, determine if it is relevant to answering the question.

Question: {question}
Contexts:
{context_list}

Output ONLY valid JSON (no markdown): {{"chunks": [{{"chunk": 1, "relevant": true/false}}], "score": 0.0-1.0}}
The score should be the fraction of chunks that are relevant."""
        result = self.llm.invoke(prompt)
        return self._parse_score(result.content)

    def context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """Did we retrieve all necessary context? (0.0 - 1.0)"""
        context_text = "\n".join(contexts)
        prompt = f"""Extract key claims from the ground truth answer. Check what fraction is covered by the contexts.

Ground Truth: {ground_truth}
Contexts: {context_text[:2000]}

Output ONLY valid JSON (no markdown): {{"claims": [{{"claim": "...", "covered": true/false}}], "score": 0.0-1.0}}
The score should be the fraction of ground truth claims covered by the contexts."""
        result = self.llm.invoke(prompt)
        return self._parse_score(result.content)

    def evaluate(self, question: str, answer: str, contexts: list[str],
                 ground_truth: str | None = None) -> dict:
        context_text = "\n".join(contexts)
        scores = {
            "faithfulness": self.faithfulness(answer, context_text),
            "answer_relevance": self.answer_relevance(answer, question),
            "context_precision": self.context_precision(contexts, question),
        }
        if ground_truth:
            scores["context_recall"] = self.context_recall(contexts, ground_truth)
        scores["overall"] = round(sum(scores.values()) / len(scores), 3)
        return scores
