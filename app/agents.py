import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from core.retriever import SemanticRetriever
from .state import AgentState
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def query_transformer_agent(state):
    print("---NODE: QUERY TRANSFORMER---")
    question = state["question"]

    prompt_template = """
    Você é um especialista em reescrever perguntas para sistemas de busca. 
    Gere 3 versões alternativas da pergunta do usuário para melhorar a recuperação de documentos. 
    Retorne as 3 perguntas como uma lista de strings em formato JSON:

    Exemplo: {{"queries": ["pergunta 1", "pergunta 2", "pergunta 3"]}}

    Pergunta Original: "{question}"
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({"question": question})
    generated_queries = result.get("queries", [])

    all_queries = [question] + generated_queries
    print(f"Perguntas geradas: {all_queries}")
    
    return {"transformed_queries": all_queries, "question": question}


def retriever_agent(state: AgentState):
    print("---NODE: RETRIEVER---")
    queries = state.get("transformed_queries", [state["question"]])
    retriever = SemanticRetriever()
    all_results = []

    for q in queries:
        results = retriever.search(q, k=5)
        all_results.extend(results)

    seen_contents = set()
    unique_results = []
    for r in all_results:
        content_hash = hash(r.document.page_content)
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_results.append(r)

    top_results = sorted(unique_results, key=lambda r: r.score, reverse=True)[:5]
    documents = [r.document for r in top_results]

    print(f"Documentos selecionados (top 5 únicos por conteúdo): {len(documents)}")
    for r in top_results:
        print(f"Score: {r.score:.3f} | Fonte: {r.document.metadata.get('source')}")
        print(r.document.page_content[:300])
        print("-" * 50)

    return {"documents": documents, "question": state["question"]}

def answerer_agent(state: AgentState):
    print("---NODE: ANSWERER---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(
        f"Fonte: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}\nConteúdo: {doc.page_content}"
        for doc in documents
    )

    prompt_template = """
    Você é um assistente especializado em Direito Previdenciário. 
    Responda à pergunta do usuário com base EXCLUSIVAMENTE nos documentos fornecidos.

    **Regras Estritas:**
    1. Cite suas fontes obrigatoriamente: use o formato (Fonte: nome_do_arquivo, Página: X).
    2. Se a informação não estiver nos documentos, responda EXATAMENTE: 
       "A informação solicitada não foi encontrada nos documentos disponíveis."
    3. Não invente informações ou faça suposições.
    4. Seja claro, conciso e direto ao ponto.

    **Contexto (Documentos):**
    {context}

    **Pergunta do Usuário:**
    {question}

    **Resposta:**
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {"answer": response.content}


def self_check_agent(state: AgentState):
    print("---(NÓ: SELF-CHECK)---")
    answer = state["answer"]
    
    if "A informação solicitada não foi encontrada" in answer:
        print("Resposta indica que não há informação. Verificação passou.")
        return {"checked_answer": answer}

    if answer and "(Fonte:" in answer:
        print("Verificação básica passou: resposta contém citações.")
        return {"checked_answer": answer}
    else:
        print("Verificação falhou: resposta sem citações. Retornando resposta padrão.")
        return {
            "checked_answer": "A resposta gerada não pôde ser validada com as fontes. A informação pode não estar presente nos documentos indexados."
        }


def safety_policy_agent(state: AgentState):
    print("---(NÓ: SAFETY/POLICY)---")
    checked_answer = state["checked_answer"]
    
    disclaimer = "\n\n**Aviso Legal:** Esta resposta é gerada por uma inteligência artificial e baseia-se nos documentos fornecidos. Não constitui aconselhamento jurídico. Consulte sempre um profissional qualificado ou as fontes oficiais do governo para tomar decisões."
    
    final_answer = checked_answer + disclaimer
    
    return {"checked_answer": final_answer}
