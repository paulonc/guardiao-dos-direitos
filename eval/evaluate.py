import os
import time
import json
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

from .generate_report import generate_report
from app.graph import build_graph


METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

def setup_models():
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small", model_kwargs={'device': 'cpu'})
    api_key = os.getenv("GOOGLE_API_KEY_RAGAS")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY_RAGAS necessária")
    
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_retries=3,
        request_timeout=60
    )
    return llm, embeddings

@dataclass
class TestCase:
    question: str
    ground_truth: str

@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    latency: float
    memory_usage: float
    metrics: dict


def load_test_cases(csv_path: str) -> List[TestCase]:
    df = pd.read_csv(csv_path)
    if not {"question", "expected_answer"}.issubset(df.columns):
        raise ValueError("CSV precisa conter 'question' e 'expected_answer'")
    return [TestCase(row["question"], row["expected_answer"]) for _, row in df.iterrows()]

def collect_response(graph, case: TestCase) -> dict:
    process = psutil.Process()
    
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    result = graph.invoke({"question": case.question})
    
    latency = time.time() - start_time
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    memory_used = mem_after

    answer = result.get("final_answer", result.get("answer", ""))
    documents = result.get("documents", [])
    contexts = [
        {
            "source": doc.metadata.get("source", "N/A") if isinstance(doc.metadata, dict) else "N/A",
            "page": doc.metadata.get("page", "N/A") if isinstance(doc.metadata, dict) else "N/A",
            "content": getattr(doc, "page_content", str(doc))
        }
        for doc in documents
    ]

    return {
        "answer": answer,
        "contexts": contexts,
        "latency": latency,
        "memory_usage": memory_used
    }



def evaluate_cases(graph, test_cases: List[TestCase], sleep_sec: float = 5.0):
    llm, embeddings = setup_models()
    results: List[EvalResult] = []

    for i, case in enumerate(test_cases, 1):
        print(f"⏳ Avaliando {i}/{len(test_cases)}: {case.question[:50]}...")
        response = collect_response(graph, case)

        dataset = Dataset.from_dict({
            "question": [case.question],
            "answer": [response["answer"]],
            "contexts": [[c["content"] for c in response["contexts"]]],
            "ground_truth": [case.ground_truth]
        })

        ragas_report = evaluate(dataset, metrics=METRICS, llm=llm, embeddings=embeddings)
        ragas_df = ragas_report.to_pandas()

        scores = {m: float(ragas_df[m].iloc[0]) for m in METRIC_NAMES if m in ragas_df}
        
        results.append(EvalResult(
            question=case.question,
            answer=response["answer"],
            contexts=response["contexts"],
            ground_truth=case.ground_truth,
            latency=response["latency"],
            memory_usage=response["memory_usage"],
            metrics=scores
        ))

        time.sleep(sleep_sec)

    return results

def summarize_results(results: List[EvalResult]):
    ragas_summary = {
        m: float(np.mean([r.metrics[m] for r in results if m in r.metrics]))
        for m in METRIC_NAMES
    }
    return {
        "timestamp": datetime.now().isoformat(),
        "evaluation_summary": {
            "total_questions": len(results),
            "avg_latency_seconds": float(np.mean([r.latency for r in results])),
            "avg_memory_usage_mb": float(np.mean([r.memory_usage for r in results])),
        },
        "ragas_metrics": ragas_summary,
        "detailed_results": [asdict(r) for r in results],
    }

if __name__ == "__main__":
    graph = build_graph()
    test_cases = load_test_cases("data/tests/test_questions.csv")
    
    results = evaluate_cases(graph, test_cases, sleep_sec=60)
    report = summarize_results(results)

    os.makedirs("eval", exist_ok=True)
    with open("eval/ragas_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open("eval/ragas_report.md", "w", encoding="utf-8") as f:
        f.write(generate_report(report))

    print("✅ Relatório salvo em eval/ragas_report.md")
