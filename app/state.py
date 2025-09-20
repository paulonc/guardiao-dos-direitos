from typing import List, TypedDict, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    category: Optional[str]
    transformed_queries: Optional[List[str]]
    documents: List[Document]
    answer: str
    checked_answer: str
    final_answer: Optional[str]