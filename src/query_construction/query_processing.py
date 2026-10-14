import re
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from typing import Any, List, Dict, cast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable


def _extract_query_as_list(x: Any ) -> List[str]:
    """Extracts a query string into a list[str]"""

    if isinstance(x, str):
        return [x]
    
    if isinstance(x, dict):
        d = cast(Dict[str, Any], x)
        query_val = d.get("question") or d.get("query", "")
        return [str(query_val)] if query_val else []
        
    return [str(x)]

def _split_semicolon_string(text: Any) -> List[str]:
    """Splits string output from LLM by semicolons"""
    s = cast(str, text) if isinstance(text, str) else str(text)
    return [q.strip() for q in s.split(";") if q.strip()]

def query_chain(
    llm: Any,
    transform_flag: bool = True,
    transform_type: str = "rewrite",
    transform_prompt: str = "{query}"
) -> Runnable:
    """Builds an LCEL query transformation chain that satisfies Pyright static analysis."""
    
    fallback_chain = RunnableLambda(_extract_query_as_list)

    if not transform_flag:
        return fallback_chain

    if transform_type == "rewrite":
        rewrite_prompt = ChatPromptTemplate.from_template(transform_prompt)

        rewrite_chain = (
            rewrite_prompt 
            | llm 
            | StrOutputParser() 
            | RunnableLambda(_split_semicolon_string)
        )

        return rewrite_chain.with_fallbacks([fallback_chain])

    return fallback_chain

def get_unique_union(documents: list[list]):
    """A simple function to get the unique union of retrieved documents"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc, allowed_objects=[Document]) for doc in unique_docs]