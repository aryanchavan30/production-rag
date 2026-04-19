import logging
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain enough information, say so clearly. "
    "Always cite the source filename when referencing specific information."
)


def _build_prompt(question: str, docs: list[dict]) -> str:
    if not docs:
        return f"No relevant documents were found. Answer from general knowledge if possible.\n\nQuestion: {question}"
    context = "\n\n".join(
        f"[Source {i + 1}: {d.get('filename', d['source'])}]\n{d['text']}"
        for i, d in enumerate(docs)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def generate(question: str, docs: list[dict], llm: ChatOpenAI) -> str:
    logger.info(f"[GENERATE] Generating answer for: '{question[:80]}' ({len(docs)} docs)")
    if docs:
        for i, doc in enumerate(docs):
            logger.info(f"[GENERATE]   source[{i}] {doc.get('filename', doc['source'])}")
    else:
        logger.warning("[GENERATE] No docs — LLM will answer from general knowledge")
    prompt = _build_prompt(question, docs)
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    answer = response.content.strip()
    logger.info(f"[GENERATE] Answer generated ({len(answer)} chars)")
    return answer


async def generate_stream(
    question: str, docs: list[dict], llm: ChatOpenAI
) -> AsyncGenerator[str, None]:
    logger.info(f"[GENERATE] Streaming answer for: '{question[:80]}' ({len(docs)} docs)")
    prompt = _build_prompt(question, docs)
    token_count = 0
    async for chunk in llm.astream(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    ):
        if chunk.content:
            token_count += 1
            yield chunk.content
    logger.info(f"[GENERATE] Stream complete ({token_count} chunks)")
