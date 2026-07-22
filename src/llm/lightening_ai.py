from typing import Any, Iterator, List, Optional, cast  # Import cast
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from litai import LLM
from llm.base import BaseLLM

class LitLLM(BaseLLM):
    """Encompasses the remote LitAI routing framework for cloud API generations."""
    
    fallback_models: Optional[List[str]] = None
    max_retries: int = 3
    api_key: str = 

    @property
    def client(self) -> LLM:
        """Lazy-loads the LitAI routing client engine."""
        if not hasattr(self, "_client_instance"):
            full_model_name = f"lightning-ai/{self.model_id}" if not self.model_id.startswith("lightning-ai/") else self.model_id
            
            self._client_instance = LLM(
                # cast full_model_name to Any to satisfy Pyright's strict literal matching
                model=cast(Any, full_model_name),
                fallback_models=self.fallback_models,
                max_retries=self.max_retries,
                api_key = self.api_key
            )
        return self._client_instance

    @property
    def _llm_type(self) -> str:
        return "lightning_ai_router"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        response_text = self.client.chat(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        for chunk in self.client.chat(prompt, stream=True):
            if run_manager:
                run_manager.on_llm_new_token(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))