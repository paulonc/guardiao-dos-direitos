from typing import List, TypedDict
from langchain.schema import Document

class AgentState(TypedDict):
    question: str
    transformed_queries: List[str]
    documents: List[Document]
    answer: str
    checked_answer: str
    agent_final_answer: str