from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    provider: str = "gemini"
    api_key: str = ""
    model: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class EvalRequest(BaseModel):
    collection: str
    dataset_path: str = "./data/eval/test_dataset.json"
    provider: str = "gemini"
    api_key: str = ""
    model: str = ""


class EvalResponse(BaseModel):
    results: list[dict]
    average_scores: dict


class UploadRequest(BaseModel):
    provider: str = "gemini"
    api_key: str = ""
