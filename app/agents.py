import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from core.retriever import SemanticRetriever
from .state import AgentState
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def supervisor_agent(state: AgentState):
    print("---NODE: SUPERVISOR---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        """Você é um supervisor de um sistema de IA. Sua função é analisar a pergunta do usuário e 
        classificá-la em uma das seguintes categorias:

        1.  `pergunta_sobre_previdencia`: A pergunta é específica sobre Direito Previdenciário, 
            benefícios, aposentadoria, INSS, contribuições, pensão por morte, auxílio-doença, etc.
        2.  `meta_pergunta`: A pergunta é sobre o próprio assistente (ex: "quem é você?", "o que você faz?").
        3.  `saudacao`: O usuário está apenas dizendo "oi", "olá", "bom dia", etc.
        4.  `fora_de_topico`: A pergunta é sobre qualquer outro assunto que não seja Direito Previdenciário.

        Pergunta do Usuário: "{question}"

        Retorne um JSON com a chave "category" e o valor sendo a categoria escolhida.
        Exemplo: {{"category": "pergunta_sobre_previdencia"}}
        """
    )
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": question})
    category = result.get("category", "fora_de_topico")
    print(f"Categoria da pergunta: {category}")
    return {"category": category}

def off_topic_agent(state: AgentState):
    print("---NODE: OFF-TOPIC---")
    prompt = ChatPromptTemplate.from_template(
        """Você é o Guardião Dos Direitos.
        O usuário perguntou sobre algo fora do tema do Direito Previdenciário.
        Recuse educadamente, explicando que você só responde perguntas relacionadas ao Direito Previdenciário,
        e convide-o a perguntar algo dentro do tema."""
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({})
    return {"final_answer": generation}

def meta_agent(state: AgentState):
    print("---NODE: META---")
    prompt = ChatPromptTemplate.from_template(
        """Você é o Guardião Dos Direitos.
        O usuário está perguntando sobre você (quem é você, o que faz, etc.).
        Explique sua função de maneira clara e envolvente: você é um assistente de IA
        especializado em responder perguntas sobre Direito Previdenciário,
        usando leis, normas e informações oficiais como base."""
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({})
    return {"final_answer": generation}

def greeting_agent(state: AgentState):
    print("---NODE: GREETING---")
    prompt = ChatPromptTemplate.from_template(
        """Você é o Guardião Dos Direitos, um assistente amigável e educado.
        O usuário está apenas cumprimentando. 
        Responda de forma simpática e acolhedora, convidando-o a fazer perguntas sobre Direito Previdenciário."""
    )
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({})
    return {"final_answer": generation}

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
        return {"final_answer": answer}

    if answer and "(Fonte:" in answer:
        print("Verificação básica passou: resposta contém citações.")
        return {"final_answer": answer}
    else:
        print("Verificação falhou: resposta sem citações. Retornando resposta padrão.")
        return {
            "final_answer": "A resposta gerada não pôde ser validada com as fontes. A informação pode não estar presente nos documentos indexados."
        }


def safety_policy_agent(state: AgentState):
    print("---(NÓ: SAFETY/POLICY)---")
    final_answer = state["final_answer"]
    
    disclaimer = "\n\n**⚠️ Aviso Legal:** Esta resposta é gerada por uma inteligência artificial e baseia-se nos documentos fornecidos. Não constitui aconselhamento jurídico. Consulte sempre um profissional qualificado ou as fontes oficiais do governo para tomar decisões."
    
    final_answer = final_answer + disclaimer
    
    return {"final_answer": final_answer}
