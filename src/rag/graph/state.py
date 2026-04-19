from typing import TypedDict


class RAGState(TypedDict):
    question: str
    expanded_queries: list[str]
    documents: list[dict]
    rewrite_count: int
