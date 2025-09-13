from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents import (
    query_transformer_agent,
    retriever_agent,
    answerer_agent,
    self_check_agent,
    safety_policy_agent,
)

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("query_transformer", query_transformer_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("answerer", answerer_agent)
    workflow.add_node("self_check", self_check_agent)
    workflow.add_node("final_answer", safety_policy_agent)

    workflow.set_entry_point("query_transformer")
    workflow.add_edge("query_transformer", "retriever")
    workflow.add_edge("retriever", "answerer")
    workflow.add_edge("answerer", "self_check")
    workflow.add_edge("self_check", "final_answer")
    workflow.add_edge("final_answer", END)

    app = workflow.compile()
    return app
