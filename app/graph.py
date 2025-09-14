from langgraph.graph import END, StateGraph
from .state import AgentState
from .agents import (
    supervisor_agent,
    query_transformer_agent,
    retriever_agent,
    answerer_agent,
    self_check_agent,
    safety_policy_agent,
    off_topic_agent,
    meta_agent,
    greeting_agent
)

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("query_transformer", query_transformer_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("answerer", answerer_agent)
    workflow.add_node("self_check", self_check_agent)
    workflow.add_node("final_answer_rag", safety_policy_agent)
    workflow.add_node("off_topic", off_topic_agent)
    workflow.add_node("meta", meta_agent)
    workflow.add_node("greeting", greeting_agent)

    workflow.set_entry_point("supervisor")

    workflow.add_edge("query_transformer", "retriever")
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", "self_check")
    workflow.add_edge("self_check", "final_answer_rag")
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["category"],
        {
            "pergunta_sobre_previdencia": "query_transformer",
            "fora_de_topico": "off_topic",
            "meta_pergunta": "meta",
            "saudacao": "greeting",
        },
    )

    workflow.add_edge("final_answer_rag", END)
    workflow.add_edge("off_topic", END)
    workflow.add_edge("meta", END)
    workflow.add_edge("greeting", END)
    
    app = workflow.compile()
    return app