import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from database.base import BaseVectorDB

class Langchainfaiss(BaseVectorDB):

    def __init__(self, embedding_func, persist_dir: str, dimension: int = 1536):
        self.persist_dir = persist_dir
        self.embedding_func = embedding_func
        
        # Check if a saved index already exists on disk
        if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
            print(f"Loading existing FAISS index from {persist_dir}...")
            self.client = FAISS.load_local(
                persist_dir, 
                embedding_func, 
                allow_dangerous_deserialization=True
            )
        else:
            print("Initializing new in-memory FAISS index...")
            self.client = FAISS(
                embedding_function=embedding_func,
                index=faiss.IndexFlatL2(dimension), #type: ignore
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )

    def ingest_batch_documents(self, batch):
        result = self.client.add_documents(batch)
        
        # Automatically persist to disk after ingestion if persist_dir is provided
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client.save_local(self.persist_dir)
            
        return result
    
    def retrieve_data(self, k: int = 3):
        return self.client.as_retriever(search_kwargs={"k": k})