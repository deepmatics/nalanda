import os
from typing import Any, Dict
import torch
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from src.ingestion.splitter.recursive import LangchainRecursive
from src.ingestion.loader.json_data import OpenRagBenchJSON
from src.retrieval.embeddings.langchain import LangchainEmbeddingModel
from src.database import LangchainChroma
from src.utils.config_loader import YamlFile
from src.llm import LitLLM
from src.query_construction import query_chain, get_unique_union
from operator import itemgetter

rag_config = YamlFile.load("./config/rag.yaml")
prompts_template = YamlFile.load("./src/prompts/templates.yaml")

INPUT_DATA_PATH = rag_config["input_data"]["path"]
EMBEDDING_MODEL = os.path.expanduser(rag_config["hf"]["embeddings"])
LLM_MODEL = rag_config["hf"]["llm_model"] 
DB_BATCH_SIZE = rag_config["vector_db"]["batch_size"]
DB_PERSIST_DIR = rag_config["vector_db"]["persist_dir"]

SYSTEM_PROMPT = prompts_template["system"]

def run_ingestion(vectorstore: LangchainChroma, final_splits: list) -> None:
    """Ingests split documents into Chroma vectorstore in batches."""
    with tqdm(total=len(final_splits), desc="Indexing chunks", unit="chunks") as pbar:
        for i in range(0, len(final_splits), DB_BATCH_SIZE):
            batch = final_splits[i : i + DB_BATCH_SIZE]
            vectorstore.ingest_batch_documents(batch)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            pbar.update(len(batch))

if __name__ == "__main__":

    embedding = LangchainEmbeddingModel(model_name=EMBEDDING_MODEL)
    vectorstore = LangchainChroma(
        embedding_func=embedding,
        persist_dir=DB_PERSIST_DIR
    )

    docs = OpenRagBenchJSON.load(INPUT_DATA_PATH)
    chunker = LangchainRecursive(rag_config["chunking"])
    final_splits = chunker.split(docs)
    
    run_ingestion(vectorstore, final_splits)

    bm25_corpus = [doc.page_content for doc in final_splits]
    llm = LitLLM(model_id=LLM_MODEL)
    retriever = vectorstore.retrieve_data(k=3)
    

    user_query = "In what areas do syllabic embeddings show potential for improvement based on current research findings?"
    updated_query = query_chain(user_query, llm)

    retrieval_chain = updated_query | retriever.map() | get_unique_union

    docs = retrieval_chain.invoke({"question": user_query})
    print(f"Total unique documents retrieved: {len(docs)}")

    sys_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    generation_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")} 
    | sys_prompt
    | llm
    | StrOutputParser()
    )

    generation_chain.invoke({"question": user_query})