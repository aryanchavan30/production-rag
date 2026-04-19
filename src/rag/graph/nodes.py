import logging
from functools import partial

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from rag.graph.state import RAGState
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.reranker import BGEReranker

logger = logging.getLogger(__name__)


def make_expand_node(llm: ChatOpenAI, n_extra: int = 2):
    def expand_queries(state: RAGState) -> dict:
        question = state["question"]
        logger.info(f"[EXPAND] Original question: '{question}'")
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
        expanded = [question] + extras
        logger.info(f"[EXPAND] Generated {len(expanded)} queries:")
        for i, q in enumerate(expanded):
            logger.info(f"[EXPAND]   [{i}] {q}")
        return {"expanded_queries": expanded}
    return expand_queries


def make_retrieve_node(retriever: HybridRetriever, reranker: BGEReranker, top_k: int = 5):
    def retrieve(state: RAGState) -> dict:
        queries = state.get("expanded_queries") or [state["question"]]
        logger.info(f"[RETRIEVE] Running {len(queries)} query/queries")
        merged: dict[str, dict] = {}
        for q in queries:
            for doc in retriever.retrieve(q):
                doc_id = f"{doc['source']}::{doc['chunk_index']}"
                if doc_id not in merged:
                    merged[doc_id] = doc
        logger.info(f"[RETRIEVE] Merged pool: {len(merged)} unique docs")
        reranked = reranker.rerank(state["question"], list(merged.values()), top_k=top_k)
        logger.info(f"[RETRIEVE] After rerank: {len(reranked)} docs")
        for i, doc in enumerate(reranked):
            logger.info(f"[RETRIEVE]   [{i}] {doc.get('filename', doc['source'])} | rerank={doc.get('rerank_score', 0):.4f} | '{doc['text'][:60]}...'")
        return {"documents": reranked}
    return retrieve


def make_grade_node(llm: ChatOpenAI):
    def grade_documents(state: RAGState) -> dict:
        question = state["question"]
        original_docs = state["documents"]
        logger.info(f"[GRADE] Grading {len(original_docs)} docs for: '{question[:60]}'")
        relevant = []
        for i, doc in enumerate(original_docs):
            prompt = (
                f"Is this document relevant to the question? Answer YES or NO only.\n"
                f"Question: {question}\n"
                f"Document: {doc['text'][:400]}"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            verdict = resp.content.strip().upper()
            is_relevant = "yes" in verdict.lower()
            logger.info(f"[GRADE]   doc[{i}] {doc.get('filename', doc['source'])} → {verdict} {'✓' if is_relevant else '✗'}")
            if is_relevant:
                relevant.append(doc)

        if relevant:
            logger.info(f"[GRADE] {len(relevant)}/{len(original_docs)} docs kept")
        else:
            logger.warning(f"[GRADE] 0/{len(original_docs)} docs passed — falling back to all reranked docs")
        return {"documents": relevant if relevant else original_docs}
    return grade_documents


def make_rewrite_node(llm: ChatOpenAI):
    def rewrite_query(state: RAGState) -> dict:
        old_question = state["question"]
        logger.info(f"[REWRITE] Attempt {state['rewrite_count'] + 1} — rewriting: '{old_question}'")
        prompt = (
            f"Rewrite this search question to be more specific and improve retrieval.\n"
            f"Original: {old_question}\n"
            f"Rewritten question (one line only):"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        new_question = resp.content.strip().splitlines()[0].strip()
        logger.info(f"[REWRITE] '{old_question}' → '{new_question}'")
        return {
            "question": new_question,
            "expanded_queries": [new_question],
            "rewrite_count": state["rewrite_count"] + 1,
        }
    return rewrite_query


def decide_next(state: RAGState, max_rewrites: int = 2) -> str:
    if state["documents"]:
        logger.info(f"[DECIDE] {len(state['documents'])} relevant docs found → done")
        return "done"
    if state["rewrite_count"] >= max_rewrites:
        logger.warning(f"[DECIDE] No relevant docs after {state['rewrite_count']} rewrite(s) — giving up → done")
        return "done"
    logger.info(f"[DECIDE] No relevant docs, rewrite_count={state['rewrite_count']} → rewrite")
    return "rewrite"
