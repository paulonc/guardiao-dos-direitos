import os
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers.cross_encoder import CrossEncoder
from functools import lru_cache

VECTOR_STORE_PATH = "data/vector_store"
EMBEDDING_MODEL = "thenlper/gte-small"
RERANKER_MODEL = "BAAI/bge-reranker-base"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int
    metadata: Dict[str, Any]


@lru_cache(maxsize=1)
def get_embeddings(model_name: str):
    logger.info(f"Carregando embeddings: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    logger.info(f"Carregando reranker: {model_name}")
    return CrossEncoder(model_name)


class SemanticRetriever:
    def __init__(
        self,
        vector_store_path: str = VECTOR_STORE_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

        self._vector_store = None

    @property
    def embeddings(self):
        return get_embeddings(self.embedding_model)

    @property
    def vector_store(self):
        if self._vector_store is None:
            logger.info("Carregando vector store FAISS...")
            self._vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self._vector_store

    @property
    def reranker(self):
        return get_reranker(self.reranker_model)

    def search(self, query: str, k: int = 5, k_retriever: int = 20) -> List[SearchResult]:
        start_time = time.time()
        logger.info(f"Consulta recebida: {query}")

        try:
            logger.info(f"Fase 1: buscando {k_retriever} candidatos com FAISS (MMR)...")
            faiss_docs = self.vector_store.max_marginal_relevance_search(
                query, k=k_retriever, fetch_k=k_retriever * 2, lambda_mult=0.5
            )
        except Exception as e:
            logger.error(f"Erro na busca FAISS: {e}")
            return []

        candidate_docs = faiss_docs

        if not candidate_docs:
            logger.warning("Nenhum documento encontrado.")
            return [
                SearchResult(
                    document=Document(page_content="Nenhum resultado encontrado."),
                    score=0.0,
                    rank=1,
                    metadata={"query": query, "retrieval_method": "empty"},
                )
            ]

        logger.info(f"Fase 2: reranqueando {len(candidate_docs)} candidatos com {self.reranker_model}...")
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(pairs, convert_to_numpy=True)

        doc_scores = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)

        results = [
            SearchResult(
                document=doc,
                score=float(score),
                rank=i + 1,
                metadata={"query": query, "retrieval_method": "reranked"},
            )
            for i, (doc, score) in enumerate(doc_scores[:k])
        ]

        elapsed = time.time() - start_time
        logger.info(f"Busca concluída em {elapsed:.2f}s | {len(results)} resultados finais")
        return results
