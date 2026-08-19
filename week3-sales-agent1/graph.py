"""
graph.py
Wires Source, Qualify, Close, and HITL into one LangGraph StateGraph
(Week 3, Skill 1, Concept 6: Agent Frameworks - LangGraph).

Design note on "the loop": each call to run_turn() is exactly ONE unit of
work - draft the opening message, react to one lead reply, or resolve one
human decision. The graph itself is a short, mostly-linear path per
invocation; what makes it feel like a continuous, resumable conversation
is the SQLite checkpointer persisting state under a thread_id (one per
lead) between separate run_turn() calls. This maps the ReAct "pause after
acting, resume when new input arrives" loop directly onto LangGraph's
native checkpoint/resume model instead of building custom pause/resume
plumbing (Concept 9: State Management & Agent Loop Control).
"""

import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import AgentState, new_lead_state
from source_agent import source_node
from qualify_agent import qualify_node
from close_agent import close_node
from hitl import hitl_node

DB_PATH = os.getenv("APEX_STATE_DB", "./apex_state.db")


def _entry_router(state: dict) -> str:
    if state.get("human_decision") is not None:
        return "hitl"
    stage = state.get("deal_stage", "New")
    if stage in ("New", ""):
        return "source"
    if stage == "Sourced":
        return "qualify"
    if stage in ("Qualified", "Engaging", "Negotiating"):
        return "close"
    return "__end__"  # Booked, Disqualified, Lost, Escalated (awaiting human) all just stop here


def _after_qualify_router(state: dict) -> str:
    return "close" if state.get("deal_stage") == "Qualified" else "__end__"


def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node("source", source_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("close", close_node)
    graph.add_node("hitl", hitl_node)

    graph.add_conditional_edges(START, _entry_router, {
        "source": "source", "qualify": "qualify", "close": "close",
        "hitl": "hitl", "__end__": END,
    })
    graph.add_edge("source", "qualify")
    graph.add_conditional_edges("qualify", _after_qualify_router, {"close": "close", "__end__": END})
    graph.add_edge("close", END)
    graph.add_edge("hitl", END)

    return graph.compile(checkpointer=checkpointer)


def get_checkpointer(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(checkpointer=get_checkpointer())
    return _compiled_graph


def run_turn(thread_id: str, lead_id: str = None, initial_lead: dict = None,
             latest_lead_message: str = None, human_decision: str = None,
             human_feedback: str = None):
    """
    The single entry point the Streamlit app (and tests) call. Handles
    whether this is a brand-new lead or a resumed conversation, based on
    whether the checkpointer already has state for this thread_id.
    """
    compiled = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = compiled.get_state(config)
    is_new = not snapshot or not snapshot.values

    if is_new:
        input_state = new_lead_state(lead_id or thread_id, initial_lead)
        input_state["latest_lead_message"] = latest_lead_message
    else:
        input_state = {"latest_lead_message": latest_lead_message}
        if human_decision is not None:
            input_state["human_decision"] = human_decision
            input_state["human_feedback"] = human_feedback

    return compiled.invoke(input_state, config=config)


def get_thread_state(thread_id: str):
    compiled = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = compiled.get_state(config)
    return snapshot.values if snapshot else None