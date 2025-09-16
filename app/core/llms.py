from typing import Any, Dict, List, Optional, Union
import httpx
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from app.core.config import settings


logger = logging.getLogger(__name__)


class DeepSeekChatModel(BaseChatModel):
    """
    Custom DeepSeek chat model implementation
    """
    
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    
    def __init__(self, **kwargs):
        # Set default values from settings
        kwargs.setdefault('api_key', settings.deepseek_api_key)
        kwargs.setdefault('base_url', settings.deepseek_base_url)
        kwargs.setdefault('model', settings.deepseek_model)
        kwargs.setdefault('temperature', settings.temperature)
        kwargs.setdefault('max_tokens', settings.max_tokens)
        
        super().__init__(**kwargs)
        
        if not self.api_key or self.api_key.startswith('your_') or self.api_key.startswith('sk-') and len(self.api_key) < 20:
            raise ValueError("Valid DeepSeek API key is required")
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response from DeepSeek API"""
        import asyncio
        return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Async generate response from DeepSeek API"""
        try:
            # Convert messages to DeepSeek format
            formatted_messages = self._format_messages(messages)
            
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": False
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
            
            # Extract response content
            content = result["choices"][0]["message"]["content"]
            
            # Create LangChain response format
            from langchain_core.outputs import ChatGeneration, ChatResult
            from langchain_core.messages import AIMessage
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to DeepSeek format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            else:
                formatted.append({"role": "assistant", "content": msg.content})
        return formatted


def _get_openai_llm() -> Optional[ChatOpenAI]:
    """Get OpenAI LLM as fallback"""
    if not settings.openai_api_key:
        logger.error("No API key available for any LLM provider")
        return None
    
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )


def get_llm() -> Union[ChatOpenAI, DeepSeekChatModel, None]:
    """Get the configured LLM instance"""
    
    if settings.llm_provider == "deepseek":
        if not settings.deepseek_api_key:
            logger.warning("DeepSeek API key not provided, falling back to OpenAI")
            return _get_openai_llm()
        
        try:
            return DeepSeekChatModel()
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek LLM: {e}")
            return _get_openai_llm()
    
    elif settings.llm_provider == "openai":
        return _get_openai_llm()
    
    else:
        logger.warning(f"Unsupported LLM provider: {settings.llm_provider}, using OpenAI")
        return _get_openai_llm()


