from langchain_huggingface import HuggingFaceEmbeddings
from src.retrieval.embeddings.base import BaseEmbeddingModel
from src.utils.device import get_device

class LangchainEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):
        """
        Initialize the HuggingFace embedding model via LangChain.
        """
        # We store the actual model instance in self.model
        self.model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": get_device()})
    
    def load(self):
        """
        Returns the internal LangChain object.
        """
        return self.model