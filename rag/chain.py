"""
rag/chain.py
------------
Assembles the full RAG chain:
  condense history → retrieve docs → generate answer

Uses only langchain_core primitives (stable across langchain 0.2/0.3/1.x).
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag.prompts import make_condense_prompt, make_qa_prompt

# MMR retriever settings
MMR_K = 8
MMR_FETCH_K = 20


def build_rag_chain(vectorstore, llm) -> Any:
    """
    Build the two-step RAG chain.

    Step 1 — condense_chain:
        Collapses (chat_history + input) → standalone search query.
        Skipped when chat_history is empty (uses raw input directly).

    Step 2 — rag_chain:
        Retrieves k=8 docs via MMR, formats context, generates answer.

    Args:
        vectorstore: FAISS (or any LangChain) vectorstore with .as_retriever().
        llm: A LangChain chat model (e.g. ChatGoogleGenerativeAI).

    Returns:
        A LangChain Runnable that accepts:
            {"input": str, "chat_history": list[BaseMessage]}
        and returns:
            {"answer": str, "context": list[Document]}
    """
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
    )

    condense_prompt = make_condense_prompt()
    qa_prompt = make_qa_prompt()
    condense_chain = condense_prompt | llm | StrOutputParser()

    def get_context(inputs: dict) -> tuple[str, list]:
        """Retrieve relevant docs; condense history into query if present."""
        if inputs.get("chat_history"):
            query = condense_chain.invoke(inputs)
        else:
            query = inputs["input"]
        docs = base_retriever.invoke(query)
        context_str = "\n\n".join(d.page_content for d in docs)
        return context_str, docs

    def build_chain_input(inputs: dict) -> dict:
        context_str, context_docs = get_context(inputs)
        return {
            "context": context_str,
            "chat_history": inputs.get("chat_history", []),
            "input": inputs["input"],
            "_context_docs": context_docs,
        }

    rag_chain = RunnableLambda(build_chain_input) | {
        "answer": (
            RunnableLambda(
                lambda x: {
                    "context": x["context"],
                    "chat_history": x["chat_history"],
                    "input": x["input"],
                }
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        ),
        "context": RunnableLambda(lambda x: x["_context_docs"]),
    }

    return rag_chain


def build_llm(temperature: float = 0.1):
    """Instantiate the Gemini 2.5 Flash LLM."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)
