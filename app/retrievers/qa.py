from typing import Any, Dict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.core.llms import get_llm


prompt = PromptTemplate.from_template(
    """
You are a helpful assistant for a vision-aware RAG system. Use the context to answer.
If the answer is not present, say you don't know.

Context:
{context}

Question: {question}

Answer:
"""
)


def answer_from_context(context: str, question: str) -> str:
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.invoke({"context": context, "question": question})
    return res["text"].strip() if isinstance(res, dict) else getattr(res, "content", str(res))


