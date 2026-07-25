import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils.config_loader import YamlFile

prompts_template = YamlFile.load("./nalanda/src/prompts/rag.yaml")
rag_config = YamlFile.load("./nalanda/config/rag.yaml")

TRANSFORM_QUERY_FLAG = rag_config["query_settings"]["use"]
TRANSFORM_QUERY_TYPE = rag_config["query_settings"]["type"]
TRANSFORM_PROMPT = prompts_template["query_transform"]

def transform(original_query: str, llm) -> str:
    """Conditionally rewrites the query using the LLM based on query transformation config flags."""
    if not TRANSFORM_QUERY_FLAG:
        return original_query

    if TRANSFORM_QUERY_TYPE == "rewrite":
        # FIX 1: Generate the prompt directly from the template string 
        # since it already contains the instruction and the {question} token.
        rewrite_prompt = ChatPromptTemplate.from_template(TRANSFORM_PROMPT)
        
        chain = rewrite_prompt | llm | StrOutputParser()
        
        try:
            raw_output = chain.invoke({"question": original_query})
            
            # Post-processing: Strip out prompt leaks or system wrappers if present
            processed = raw_output
            if "<|im_start|>assistant" in processed:
                processed = processed.split("<|im_start|>assistant")[-1]
            
            # Clean out reasoning blocks if using a thinking/reasoning model
            if "</think>" in processed:
                processed = processed.split("</think>")[-1]
            else:
                processed = re.sub(r'<think>.*?</think>', '', processed, flags=re.DOTALL)
            
            optimized_query = processed.replace("<|im_end|>", "").strip()
            
            # FIX 3: Print separate variables so you can accurately audit changes in logs
            print(f"\n🔄 [Query Transform]")
            print(f"   Original:  '{original_query}'")
            print(f"   👉 DB Query: '{optimized_query}'\n")
            
            return optimized_query
            
        except Exception as e:
            print(f"⚠️ [Query Transform Error] Fallback to original due to exception: {e}")
            return original_query

    return original_query


original_query = "Why is the sky blue and the water also blue?"