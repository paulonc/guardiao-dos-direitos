import json
from datetime import datetime

def quote_text(text: str) -> str:
    return "\n".join([f"> {line}" for line in text.splitlines()])

def get_score_indicator(score: float, lower_is_better: bool = False) -> str:
    if score is None:
        return "⚪️"

    if not lower_is_better:
        if score >= 0.8: return "🟢"
        if score >= 0.6: return "🟡"
        return "🔴"
    else:
        if score <= 2.0: return "🟢"
        if score <= 5.0: return "🟡"
        return "🔴"


def generate_report(report: dict) -> str:
    summary = report.get("evaluation_summary", {})
    ragas_metrics = report.get("ragas_metrics", {})
    detailed_results = report.get("detailed_results", [])

    md = [
        "# 📈 Relatório de Avaliação RAG",
        f"> Gerado em: {report.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
        ""
    ]

    md.append("## 📊 Métricas Agregadas")
    md.append(f"Baseado em **{summary.get('total_questions', 0)}** questões**")
    md.append("")
    md.append("| Métrica | Média | Indicador |")
    md.append("| :--- | :---: | :---: |")
    for metric, value in ragas_metrics.items():
        md.append(f"| **{metric.replace('_',' ').title()}** | `{value:.3f}` | {get_score_indicator(value)} |")
    md.append(f"| **Latência Média (s)** | `{summary.get('avg_latency_seconds', 0):.3f}` | {get_score_indicator(summary.get('avg_latency_seconds'), lower_is_better=True)} |")
    md.append(f"| **Memória Média (MB)** | `{summary.get('avg_memory_usage_mb', 0):.2f}` | N/A |")
    md.append("---")
    md.append("")

    md.append("## 🔬 Detalhes por Pergunta")
    for i, r in enumerate(detailed_results, 1):
        md.append(f"### {i}. Pergunta")
        md.append(f"{r.get('question', 'N/A')}")
        md.append("")

        md.append("**Scores por Pergunta:**")
        metrics = r.get("metrics", {})
        scores_line = []
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = metrics.get(metric)
            if val is not None:
                scores_line.append(f"{metric.replace('_',' ').title()}: `{val:.3f}` {get_score_indicator(val)}")
            else:
                scores_line.append(f"{metric.replace('_',' ').title()}: N/A")
        md.append(" | ".join(scores_line))
        md.append("")


        md.extend([
            "<details>",
            "<summary><strong>🤖 Resposta Gerada vs. ✅ Ground Truth</strong></summary>",
            "",
            "**Resposta Gerada:**",
            quote_text(r.get('answer', 'N/A')),
            "",
            "**Ground Truth:**",
            f"> {r.get('ground_truth', 'N/A')}",
            "</details>",
            ""
        ])

        md.extend([
            "<details>",
            "<summary><strong>📚 Contextos Recuperados</strong></summary>",
            ""
        ])
        for c in r["contexts"]:
            if isinstance(c, dict):
                md.append(f"- **Fonte:** `{c.get('source', 'N/A')}` (Página: {c.get('page', 'N/A')})")
                md.append(quote_text(c.get('content', '')))
            else:
                md.append(f"- {c[:300]}...")

        md.append("</details>")
        md.append("")
        md.append("---")    
        md.append("")
    return "\n".join(md)

if __name__ == "__main__":
    with open("eval/ragas_report.json", "r", encoding="utf-8") as f:
        report_data = json.load(f)

    markdown_content = generate_report(report_data)

    output_file = "eval/ragas_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ Relatório markdown gerado: {output_file}")
