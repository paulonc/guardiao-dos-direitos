import sys
import os
import nest_asyncio
import streamlit as st

nest_asyncio.apply()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph import build_graph
from app.state import AgentState

st.set_page_config(
    page_title="Guardião dos Direitos",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Guardião dos Direitos")
st.caption("Assistente jurídico com LangGraph, Gemini e FAISS para consultas jurídicas")

@st.cache_resource
def load_guardiao_graph():
    try:
        graph = build_graph()
        return graph
    except Exception as e:
        st.error(f"Erro ao inicializar o Guardião: {e}")
        st.stop()

graph = load_guardiao_graph()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Em que posso ajudar com suas dúvidas jurídicas?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta sobre os documentos jurídicos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""
        
        try:
            inputs: AgentState = {"question": prompt, "context": [], "answer": "", "final_answer": ""}
            final_state = graph.invoke(inputs)
            response_text = final_state.get("checked_answer", "Não foi possível gerar uma resposta.")
            message_placeholder.markdown(response_text)
        except Exception as e:
            st.error(f"Erro ao processar a consulta: {e}")
            response_text = "Desculpe, ocorreu um erro ao processar sua solicitação."
            message_placeholder.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
