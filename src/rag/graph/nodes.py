from functools import partial
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rag.graph.state import RAGState
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.reranker import BGEReranker


def make_expand_node(llm: ChatOpenAI, n_extra: int = 2):
    def expand_queries(state: RAGState) -> dict:
        question = state["question"]
        prompt = (
            f"Generate {n_extra} alternative phrasings of this search question "
            f"to improve document retrieval.\n"
            f"Original question: {question}\n"
            f"Output one question per line, no numbering, no bullets, no explanation."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        extras = [
            line.strip()
            for line in response.content.strip().splitlines()
            if line.strip()
        ][:n_extra]
        return {"expanded_queries": [question] + extras}
    return expand_queries


def make_retrieve_node(retriever: HybridRetriever, reranker: BGEReranker, top_k: int = 5):
    def retrieve(state: RAGState) -> dict:
        queries = state.get("expanded_queries") or [state["question"]]
        merged: dict[str, dict] = {}
        for q in queries:
            for doc in retriever.retrieve(q):
                doc_id = f"{doc['source']}::{doc['chunk_index']}"
                if doc_id not in merged:
                    merged[doc_id] = doc
        reranked = reranker.rerank(state["question"], list(merged.values()), top_k=top_k)
        return {"documents": reranked}
    return retrieve


def make_grade_node(llm: ChatOpenAI):
    def grade_documents(state: RAGState) -> dict:
        question = state["question"]
        relevant = []
        for doc in state["documents"]:
            prompt = (
                f"Is this document relevant to the question? Answer YES or NO only.\n"
                f"Question: {question}\n"
                f"Document: {doc['text'][:400]}"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            if "yes" in resp.content.lower():
                relevant.append(doc)
        return {"documents": relevant}
    return grade_documents


def make_rewrite_node(llm: ChatOpenAI):
    def rewrite_query(state: RAGState) -> dict:
        prompt = (
            f"Rewrite this search question to be more specific and improve retrieval.\n"
            f"Original: {state['question']}\n"
            f"Rewritten question (one line only):"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        new_question = resp.content.strip().splitlines()[0].strip()
        return {
            "question": new_question,
            "expanded_queries": [new_question],
            "rewrite_count": state["rewrite_count"] + 1,
        }
    return rewrite_query


def decide_next(state: RAGState, max_rewrites: int = 2) -> str:
    if state["documents"]:
        return "done"
    if state["rewrite_count"] >= max_rewrites:
        return "done"
    return "rewrite"
