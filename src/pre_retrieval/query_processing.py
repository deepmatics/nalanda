import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils.config_loader import YamlFile
from langchain_core.load import dumps, loads

prompts_template = YamlFile.load("./src/prompts/templates.yaml")
rag_config = YamlFile.load("./config/rag.yaml")

TRANSFORM_QUERY_FLAG = rag_config["query_settings"]["use"]
TRANSFORM_QUERY_TYPE = rag_config["query_settings"]["type"]
TRANSFORM_PROMPT = prompts_template["multi_query"]

def transform_query(original_query: str, llm) -> list:
    """Conditionally rewrites the query using the LLM based on query transformation config flags."""

    if not TRANSFORM_QUERY_FLAG:
        return [original_query]

    if TRANSFORM_QUERY_TYPE == "rewrite":

        rewrite_prompt = ChatPromptTemplate.from_template(TRANSFORM_PROMPT)

        chain = (
                rewrite_prompt 
                | llm 
                | StrOutputParser() 
                |(lambda x: x.split(";")))
        
        try:
            transformed_output = chain.invoke({"question": original_query})
            
            return transformed_output
            
        except Exception as e:
            print(f"⚠️ [Query Transform Error] Fallback to original due to exception: {e}")
            return [original_query]

    return [original_query]

def get_unique_union(documents: list[list]):
    """ A simple function to get the unique union of retrieved documents """
    # Flatten the list of lists and convert each Document to a string for uniqueness
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]
    