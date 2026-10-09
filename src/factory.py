#TODO: Introduce Lazy Loading
from typing import Any

_LLM_REGISTRY = ["litllm", "huggingface"]

def get_llm(provider: str, **kwargs: Any) -> Any:
    """Dynamically fetches and instantiates the requested LLM provider."""
    provider = provider.lower()
    
    if provider == "litllm":
        from src.llm import LitLLM
        return LitLLM(**kwargs)
        
    elif provider == "huggingface":
        from src.llm import HuggingFaceLLM
        return HuggingFaceLLM(**kwargs)
        
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose from: {_LLM_REGISTRY}")