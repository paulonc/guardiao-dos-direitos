import os
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

VECTOR_STORE_PATH = "data/vector_store"
EMBEDDING_MODEL = "thenlper/gte-small"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int
    metadata: Dict[str, Any]

class SemanticRetriever:
    
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH, embedding_model: str = EMBEDDING_MODEL):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self._embeddings = None
        self._vector_store = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info("Carregando embeddings...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        return self._embeddings
    
    @property
    def vector_store(self):
        if self._vector_store is None:
            logger.info("Carregando vector store FAISS...")
            self._vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self._vector_store
    
    def _get_query_embedding(self, query: str):
        return np.array(self.embeddings.embed_query(query), dtype=np.float32)
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        start = time.time()
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        results = []
        for i, (doc, score) in enumerate(docs_and_scores):
            sim_score = 1 / (1 + score)
            if sim_score < score_threshold:
                continue
            results.append(SearchResult(document=doc, score=sim_score, rank=i+1, metadata={"query": query}))
        logger.info(f"Busca concluída em {time.time() - start:.2f}s | {len(results)} resultados")
        return results


if __name__ == "__main__":
    retriever = SemanticRetriever()
    queries = ["aposentadoria por idade", "auxílio-doença"]
    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = retriever.search(q, k=3)
        for r in results:
            print(f"Rank {r.rank} | Score {r.score:.3f} | Fonte: {r.document.metadata.get('source', 'N/A')}")
            print(f"Preview: {r.document.page_content[:150]}...\n")
