import torch
from threading import Thread
from typing import Any, Iterator, List, Optional
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from llm.base import BaseLLM
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

class HuggingFaceLLM(BaseLLM):
    """Downloads and runs open-source weights directly from Hugging Face."""
    
    device_map: str = "auto"
    torch_dtype: str = "float16"
    max_new_tokens: int = 512

    @property
    def model_pipeline(self) -> Any:
        """Lazy-loads and downloads the HF model onto local hardware execution context."""
        if not hasattr(self, "_pipeline_instance"):
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=self.device_map
            )
            
            self._pipeline_instance = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens
            )
        return self._pipeline_instance

    @property
    def _llm_type(self) -> str:
        return "huggingface_local"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        outputs = self.model_pipeline(prompt)
        generated_text = outputs[0]["generated_text"]
        
        # Strip out prompt from the generated completion if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=generated_text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        tokenizer = self.model_pipeline.tokenizer
        model = self.model_pipeline.model

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=self.max_new_tokens)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if run_manager:
                run_manager.on_llm_new_token(new_text)
            yield ChatGenerationChunk(message=AIMessageChunk(content=new_text))