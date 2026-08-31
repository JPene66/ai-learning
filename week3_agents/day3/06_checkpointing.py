"""
06_checkpointing.py
Implements state persistence using LangGraph's SqliteSaver.
"""

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from 05_state_schema import AgentState


# Create a persistent SQLite database
conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Build the graph with checkpointing
# (In a real implementation, you'd add your actual nodes here)
builder = StateGraph(AgentState)

# Placeholder nodes - replace with your actual implementation
def dummy_node(state: AgentState):
    return {"messages": [HumanMessage(content="Processing...")]}

builder.add_node("process", dummy_node)
builder.set_entry_point("process")
builder.add_edge("process", END)

# Compile with the checkpoint saver
graph = builder.compile(checkpointer=memory)


def run_with_checkpoint(thread_id: str, message: str):
    """Run the agent with checkpointing."""
    config = {
        "configurable": {
            "thread_id": thread_id,  # Unique per user/session
            "checkpoint_ns": ""
        },
        "recursion_limit": 25  # Safety: max 25 loop iterations
    }
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    print(f"State saved. Thread: {thread_id}")
    return result


def resume_from_checkpoint(thread_id: str):
    """Resume execution from a previous checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get the current state
    state = memory.get(config)
    print(f"Resumed from checkpoint. Last message: {state['messages'][-1].content[:100]}")
    
    # Continue execution from where it left off
    result = graph.invoke(None, config=config)  # None = resume from checkpoint
    return result


def list_checkpoints(thread_id: str):
    """List all checkpoints for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    for checkpoint in memory.list(config):
        print(f"Checkpoint at step {checkpoint['step']}: {checkpoint['ts']}")


if __name__ == "__main__":
    # Run with a thread ID
    run_with_checkpoint("session_001", "Research Tesla and build a battlecard")
    
    # List checkpoints
    list_checkpoints("session_001")
    
    # Resume (in a new process, this would pick up where it left off)
    # result = resume_from_checkpoint("session_001")