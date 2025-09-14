import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.graph import build_graph

st.set_page_config(
    page_title="Guardião dos Direitos",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Guardião dos Direitos")
st.caption("Seu assistente de IA especializado em Direito Previdenciário")

with st.sidebar:
    st.header("Opções")
    if st.button("Limpar Histórico do Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("Sobre")
    st.markdown(
        "Este assistente utiliza um sistema de agentes com LangGraph, "
        "o LLM Gemini do Google e busca vetorial com FAISS para responder "
        "suas perguntas com base em documentos jurídicos."
    )

@st.cache_resource
def load_guardiao_graph():
    try:
        return build_graph()
    except Exception as e:
        st.error(f"Erro crítico ao inicializar o Guardião: {e}")
        st.exception(e)
        st.stop()

graph = load_guardiao_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Olá! Sou o Guardião dos Direitos. Como posso ajudar com suas dúvidas sobre Direito Previdenciário hoje?")
    ]

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)
        if isinstance(message, AIMessage) and "sources" in message.additional_kwargs:
            with st.expander("Ver Fontes Utilizadas"):
                for source in message.additional_kwargs["sources"]:
                    st.info(f"**Fonte:** {source.get('source', 'N/A')}, **Página:** {source.get('page', 'N/A')}")
                    st.text(source.get('content', ''))


if prompt := st.chat_input("Qual sua dúvida sobre Direito Previdenciário?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Analisando sua pergunta...")
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            inputs = {"question": prompt}
            
            for event in graph.stream(inputs, stream_mode="values"):
                
                if "supervisor" in event:
                    status.update(label="⚖️ Classificando pergunta...")
                elif "query_transformer" in event:
                    status.update(label="✍️ Refinando a busca...")
                elif "retriever" in event:
                    status.update(label="📚 Buscando documentos relevantes...")
                elif "answerer" in event:
                    status.update(label="🧠 Gerando a resposta...")
                
                if "final_answer" in event and event["final_answer"]:
                    full_response = event["final_answer"]
                    message_placeholder.markdown(full_response)
                    if event.get("documents"):
                        sources = [
                            {"source": doc.metadata.get('source'), "page": doc.metadata.get('page'), "content": doc.page_content}
                            for doc in event["documents"]
                        ]

            status.update(label="Concluído!", state="complete")

        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
            full_response = "Desculpe, não consegui processar sua solicitação no momento."
            message_placeholder.markdown(full_response)

    ai_message = AIMessage(content=full_response, additional_kwargs={"sources": sources})
    st.session_state.messages.append(ai_message)
    
    if sources:
        with st.expander("Ver Fontes Utilizadas"):
            for source in sources:
                st.info(f"**Fonte:** {source.get('source', 'N/A')}, **Página:** {source.get('page', 'N/A')}")
                st.text(source.get('content', ''))