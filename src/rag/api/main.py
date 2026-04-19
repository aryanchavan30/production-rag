import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from rag.config import settings
from rag.generation.generator import generate_stream
from rag.graph.graph import AdaptiveRAGGraph
from rag.ingestion.embedder import Embedder
from rag.ingestion.indexer import BM25Indexer, QdrantIndexer
from rag.retrieval.reranker import BGEReranker
from rag.retrieval.retriever import HybridRetriever
from rag.api.schemas import QueryRequest


_graph: AdaptiveRAGGraph | None = None
_llm: ChatOpenAI | None = None


def _build_graph() -> tuple[AdaptiveRAGGraph, ChatOpenAI]:
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key="ollama",
        temperature=0,
    )
    embedder = Embedder(model_path=settings.embed_model_path)
    qdrant = QdrantIndexer(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        embed_dim=settings.embed_dim,
    )
    bm25 = BM25Indexer.load(settings.bm25_index_path)
    retriever = HybridRetriever(qdrant=qdrant, bm25=bm25, embedder=embedder)
    reranker = BGEReranker(model_path=settings.reranker_model_path)
    graph = AdaptiveRAGGraph(retriever=retriever, reranker=reranker, llm=llm)
    return graph, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _llm
    _graph, _llm = await asyncio.to_thread(_build_graph)
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    graph = _graph
    llm = _llm

    async def event_stream():
        state = await asyncio.to_thread(graph.run_retrieval, request.question)
        docs = state["documents"]

        sources = [
            {
                "text": d["text"][:300],
                "source": d["source"],
                "filename": d.get("filename", d["source"]),
                "rerank_score": d.get("rerank_score", 0.0),
            }
            for d in docs
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        async for token in generate_stream(state["question"], docs, llm):
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def start():
    import uvicorn
    uvicorn.run("rag.api.main:app", host="0.0.0.0", port=8000, reload=True)
