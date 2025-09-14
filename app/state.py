from typing import List, TypedDict, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    category: Optional[str]  # A categoria decidida pelo supervisor
    transformed_queries: Optional[List[str]]
    documents: List[Document]
    answer: str
    checked_answer: str
    final_answer: Optional[str] # Resposta final dos agentes de borda (off-topic, meta, etc.)