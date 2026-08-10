"""
09_hitl_graph.py
Builds the complete graph with HITL routing and checkpointing.
"""

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from 05_state_schema import AgentState
from 08_hitl_node import hitl_checkpoint_node


# Create checkpoint database
conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


# Placeholder nodes (replace with actual implementations)
def planner_node(state: AgentState):
    print("📋 Planning...")
    return {"messages": [HumanMessage(content="Plan created")]}


def executor_node(state: AgentState):
    print("⚙️ Executing...")
    return {"messages": [HumanMessage(content="Tasks executed")]}


def llm_node(state: AgentState):
    print("🧠 LLM reasoning...")
    return {"messages": [HumanMessage(content="LLM output with tool calls")]}


def tool_node(state: AgentState):
    print("🔧 Executing tools...")
    return {"messages": [HumanMessage(content="Tool result")]}


# Build the graph with HITL
def build_hitl_graph():
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("hitl_gate", hitl_checkpoint_node)
    
    # Entry point
    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "llm")
    
    # After LLM, check if HITL is needed BEFORE executing tools
    def route_after_llm(state: AgentState):
        last_msg = state["messages"][-1]
        
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return END
        
        # Check if any tool call is high-stakes
        high_stakes_tools = {"send_email", "post_slack", "save_file", "finalize_report"}
        for tc in last_msg.tool_calls:
            if tc.get("name") in high_stakes_tools:
                return "hitl_gate"
        
        return "tools"
    
    builder.add_conditional_edges("llm", route_after_llm, {
        "hitl_gate": "hitl_gate",
        "tools": "tools",
        END: END
    })
    
    # After HITL approval, proceed to tools
    builder.add_edge("hitl_gate", "tools")
    
    # After tools, loop back to LLM for next reasoning step
    builder.add_edge("tools", "llm")
    
    # Compile with checkpointing
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    graph = build_hitl_graph()
    
    config = {"configurable": {"thread_id": "session_hitl_001"}}
    
    # Start the agent
    inputs = {"messages": [HumanMessage(content="Research Tesla and email the battlecard to the team")]}
    
    # Run the graph
    result = graph.invoke(inputs, config=config)
    print("Graph execution completed")