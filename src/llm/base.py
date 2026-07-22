from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

class BaseLLM(BaseChatModel):
    """
    Abstract Base LLM wrapper for LangChain compliance.
    Subclasses must implement their own _generate and _stream methods.
    """
    model_id: str

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Converts structured LangChain messages into a single string prompt."""
        formatted = []
        for msg in messages:
            role = "User" if msg.type == "user" else "Assistant" if msg.type == "ai" else "System"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)