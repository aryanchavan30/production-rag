from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    collection: str = "rag_documents"
    top_k: int = 5


class SourceDoc(BaseModel):
    text: str
    source: str
    filename: str
    rerank_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]
