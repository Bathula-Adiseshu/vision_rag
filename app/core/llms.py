from typing import Optional

from langchain_openai import ChatOpenAI
from app.core.config import get_settings


def get_llm() -> ChatOpenAI:
    s = get_settings()
    if s.llm_provider.lower() == "deepseek":
        return ChatOpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url or "https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.2,
            timeout=60,
        )
    else:
        return ChatOpenAI(
            api_key=s.openai_api_key,
            base_url=s.openai_base_url or "https://api.openai.com/v1",
            model="gpt-4o-mini",
            temperature=0.2,
            timeout=60,
        )


