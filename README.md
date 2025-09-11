# 🛡️ Guardião dos Direitos

O **Guardião dos Direitos** é um sistema de **RAG (Retrieval-Augmented Generation) baseado em agentes**, voltado inicialmente para a área de **Direito Previdenciário**.  
Ele foi pensado para ajudar estudantes, pesquisadores e cidadãos a **consultar informações de documentos oficiais, leis e regulamentos públicos**, de forma **transparente, com citações obrigatórias e sem alucinações**.

---

## 🚀 Sobre o projeto

- Nome: **Guardião dos Direitos**  
- Objetivo: Fornecer respostas fundamentadas a partir de documentos jurídicos, sempre com **citações diretas às fontes**.  
- Tecnologias:  
  - **LangChain + LangGraph** (orquestração de agentes)  
  - **FAISS** (banco vetorial)  
  - **Embeddings abertos (GTE-Small ou BGE-Small)**  
  - **Streamlit** (interface simples e interativa)  
  - **Gemini** como modelo de linguagem principal  
  - **Docker + Makefile** para reprodutibilidade  

---

## 📖 Como funciona

1. O usuário envia uma pergunta ou carrega documentos (PDF/HTML).  
2. O sistema indexa e busca trechos relevantes usando FAISS.  
3. Um grafo de agentes decide o fluxo:
   - **Retriever Agent**: busca os documentos.  
   - **Answerer Agent**: gera a resposta com citações obrigatórias.  
   - **Self-check Agent**: valida se tudo tem evidência.  
   - **Safety Agent**: adiciona disclaimer ético.  
4. A interface exibe a resposta final, sempre com fontes.

---

## ⚠️ Aviso Importante

Este projeto é **educacional e experimental**.  
As respostas **não constituem aconselhamento jurídico**.  
Consulte sempre fontes oficiais ou um profissional qualificado.
