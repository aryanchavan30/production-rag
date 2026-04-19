import logging
from functools import partial

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from rag.graph.state import RAGState
from rag.graph.nodes import (
    make_expand_node,
    make_retrieve_node,
    make_grade_node,
    make_rewrite_node,
    decide_next,
)
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)


class AdaptiveRAGGraph:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BGEReranker,
        llm: ChatOpenAI,
        max_rewrites: int = 2,
        rerank_top_k: int = 5,
    ):
        logger.info(f"Building AdaptiveRAGGraph (max_rewrites={max_rewrites}, rerank_top_k={rerank_top_k})")
        self._max_rewrites = max_rewrites
        self._graph = self._build(retriever, reranker, llm, max_rewrites, rerank_top_k)
        logger.info("AdaptiveRAGGraph ready — flow: START → expand → retrieve → grade → [rewrite →] END")

    def _build(self, retriever, reranker, llm, max_rewrites, rerank_top_k):
        builder = StateGraph(RAGState)

        builder.add_node("expand",   make_expand_node(llm))
        builder.add_node("retrieve", make_retrieve_node(retriever, reranker, top_k=rerank_top_k))
        builder.add_node("grade",    make_grade_node(llm))
        builder.add_node("rewrite",  make_rewrite_node(llm))

        builder.add_edge(START, "expand")
        builder.add_edge("expand", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            partial(decide_next, max_rewrites=max_rewrites),
            {"rewrite": "rewrite", "done": END},
        )
        builder.add_edge("rewrite", "retrieve")

        return builder.compile()

    def run_retrieval(self, question: str) -> RAGState:
        logger.info(f"[GRAPH] Starting retrieval for: '{question}'")
        result = self._graph.invoke(
            {
                "question": question,
                "expanded_queries": [],
                "documents": [],
                "rewrite_count": 0,
            }
        )
        logger.info(f"[GRAPH] Done — rewrite_count={result['rewrite_count']}, final_docs={len(result['documents'])}")
        return result
