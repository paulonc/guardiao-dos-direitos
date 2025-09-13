import os
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder

VECTOR_STORE_PATH = "data/vector_store"
EMBEDDING_MODEL = "thenlper/gte-small"
RERANKER_MODEL = "BAAI/bge-reranker-base" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int
    metadata: Dict[str, Any]

class SemanticRetriever:
    
    def __init__(self, vector_store_path: str = VECTOR_STORE_PATH, 
                 embedding_model: str = EMBEDDING_MODEL, 
                 reranker_model: str = RERANKER_MODEL):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        
        self._embeddings = None
        self._vector_store = None
        self._reranker = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info(f"Carregando embeddings: {self.embedding_model}")
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

    @property
    def reranker(self):
        if self._reranker is None:
            logger.info(f"Carregando reranker: {self.reranker_model}")
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker
    
    def search(self, query: str, k: int = 5, k_retriever: int = 20) -> List[SearchResult]:
        start_time = time.time()
        logger.info(f"Fase 1: Buscando {k_retriever} candidatos com FAISS...")
        candidate_docs = self.vector_store.similarity_search(query, k=k_retriever)
        
        if not candidate_docs:
            logger.warning("Nenhum documento encontrado na busca inicial.")
            return []
            
        logger.info(f"Fase 2: Reordenando {len(candidate_docs)} candidatos com Cross-Encoder...")
        
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        
        scores = self.reranker.predict(pairs, convert_to_numpy=True)
        
        doc_scores = list(zip(candidate_docs, scores))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (doc, score) in enumerate(doc_scores[:k]):
            results.append(SearchResult(
                document=doc, 
                score=float(score),
                rank=i + 1, 
                metadata={"query": query, "retrieval_method": "reranked"}
            ))
            
        end_time = time.time()
        logger.info(f"Busca avançada concluída em {end_time - start_time:.2f}s | {len(results)} resultados finais")
        return results