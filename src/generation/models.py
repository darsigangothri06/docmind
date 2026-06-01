from dataclasses import dataclass


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
