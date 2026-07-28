from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    embedding_provider: str = "gemini"
    embedding_api_key: str = ""
    embedding_model: str = "models/gemini-embedding-001"

    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./data/documents"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)


def ensure_dirs(settings: Settings):
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
