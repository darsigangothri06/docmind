from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import SYSTEM_PROMPT
from .models import RAGResponse
from ..retrieval.retriever import DocumentRetriever


class RAGChain:
    """Orchestrates retrieval + generation with source citations."""

    def __init__(self, retriever: DocumentRetriever, llm):
        self.retriever = retriever
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        self.chain = (
            {"context": retriever.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question: str) -> RAGResponse:
        docs = self.retriever.retrieve(question)
        answer = self.chain.invoke(question)
        return RAGResponse(
            answer=answer,
            sources=[
                {"content": d.page_content[:200], "metadata": d.metadata}
                for d in docs
            ],
        )

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, "
            f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )


def get_llm(provider: str, api_key: str, model: str):
    """Factory to create the appropriate LLM based on provider."""
    if provider == "openai":
        return ChatOpenAI(model=model or "gpt-4o-mini", api_key=api_key)
    return ChatGoogleGenerativeAI(
        model=model or "gemini-2.5-flash", google_api_key=api_key
    )
