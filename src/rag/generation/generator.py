from typing import AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


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
    prompt = _build_prompt(question, docs)
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return response.content.strip()


async def generate_stream(
    question: str, docs: list[dict], llm: ChatOpenAI
) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(question, docs)
    async for chunk in llm.astream(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    ):
        if chunk.content:
            yield chunk.content
