import os
import torch
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.ingestion.splitter.recursive import LangchainRecursive
from src.ingestion.loader.json_data import OpenRagBenchJSON
from src.retrieval.embeddings.langchain import LangchainEmbeddingModel
from src.database import LangchainChroma
from src.utils.config_loader import YamlFile
from src.llm import LitLLM
from src.query_construction import query_chain, get_unique_union
from operator import itemgetter
import pandas as pd
from pathlib import Path

rag_config = YamlFile.load('./config/rag.yaml')
prompts_template = YamlFile.load('./src/prompts/templates.yaml')

INPUT_DATA_PATH =  rag_config['input_data']['data_path'] + 'pdf/arxiv/corpus/'
EMBEDDING_MODEL = os.path.expanduser(rag_config['hf']['embeddings'])
LLM_MODEL = rag_config['hf']['llm_model'] 
DB_BATCH_SIZE = rag_config['vector_db']['batch_size']
DB_PERSIST_DIR = rag_config['vector_db']['persist_dir']
QUERY_TRANSFORM_FLAG = rag_config['query_settings']['use']
QUERY_TRANSFORM_TYPE = rag_config['query_settings']['type']
QUERIES_SAMPLE_SIZE = rag_config['sample']['size']
RANDOM_STATE = rag_config['sample']['state']
SYSTEM_PROMPT = prompts_template['system']
QUERY_TRANSFORM_PROMPT = prompts_template['multi_query']

def run_ingestion(vectorstore: LangchainChroma, final_splits: list) -> None:
    '''Ingests split documents into Chroma vectorstore in batches.'''
    with tqdm(total=len(final_splits), desc='Indexing chunks', unit='chunks') as pbar:
        for i in range(0, len(final_splits), DB_BATCH_SIZE):
            batch = final_splits[i : i + DB_BATCH_SIZE]
            vectorstore.ingest_batch_documents(batch)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            pbar.update(len(batch))

if __name__ == '__main__':

    embedding = LangchainEmbeddingModel(model_name=EMBEDDING_MODEL)
    vectorstore = LangchainChroma(
        embedding_func=embedding,
        persist_dir=DB_PERSIST_DIR
    )

    docs = OpenRagBenchJSON.load(INPUT_DATA_PATH)
    chunker = LangchainRecursive(rag_config['chunking'])
    final_splits = chunker.split(docs)
    
    run_ingestion(vectorstore, final_splits)

    llm = LitLLM(model_id=LLM_MODEL)
    retriever = vectorstore.retrieve_data(k=3)

    queries =  pd.read_json('data/input/OpenRAGBench/pdf/arxiv/queries.json', orient='index')
    queries = queries.reset_index().rename(columns={'index': 'query_id'})
    queries = queries.loc[queries['source'] == 'text']
    queries = queries.sample(n = QUERIES_SAMPLE_SIZE, replace=False, random_state = RANDOM_STATE).reset_index(drop = True)

    extracted_answers = pd.DataFrame(columns = ["query_id", "rag_answer"])

    for i in tqdm(range(0, QUERIES_SAMPLE_SIZE), desc="RAG Retrieval"):
        user_query = queries["query"][i]
        updated_query = query_chain(llm, 
                                    QUERY_TRANSFORM_FLAG, 
                                    QUERY_TRANSFORM_TYPE,
                                    QUERY_TRANSFORM_PROMPT)

        # constructed_queries = updated_query.invoke({'question': user_query})
        # print('Constructed Queries:', constructed_queries)

        retrieval_chain = updated_query | retriever.map() | get_unique_union

        docs = retrieval_chain.invoke({'question': user_query})
        # print(f'Total unique documents retrieved: {len(docs)}')

        sys_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

        generation_chain = (
        {'context': retrieval_chain, 'question': itemgetter('question')} 
        | sys_prompt
        | llm
        | StrOutputParser()
        )
        
        answer = generation_chain.invoke({'question': user_query})
        # print(answer)

        combined_df = pd.DataFrame([[queries['query_id'][i], answer]], columns = ['query_id', "rag_answer"])
        extracted_answers = pd.concat([extracted_answers, combined_df], axis = 0)

    extracted_answers.to_csv(Path('data/output/rag_answers.csv'), index = False, encoding = 'utf-8')