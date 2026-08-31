"""
12_day3_deliverable.py
Complete Day 3 system: Planner + Executor + State Persistence + HITL
"""

import json
import sqlite3
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from 01_task_models import Task, TaskStatus
from 02_planner import planner_node
from 03_executor import execute_task
from 05_state_schema import AgentState
from 08_hitl_node import hitl_checkpoint_node


def executor_node(state: AgentState) -> dict:
    """Execute tasks in dependency order."""
    tasks = state.get("tasks", [])
    if not tasks:
        return {"messages": [AIMessage(content="No tasks to execute.")]}
    
    completed_ids = set(state.get("completed_task_ids", []))
    completed_results = {}
    
    # Find and execute ready tasks
    pending = [t for t in tasks if t.status != TaskStatus.COMPLETED]
    ready = [t for t in pending if t.is_ready(completed_ids)]
    
    if not ready:
        return {"messages": [AIMessage(content="No tasks ready to execute. Check dependencies.")]}
    
    for task in ready:
        print(f"▶️ Executing {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        
        # Get results from dependencies
        for dep in task.dependencies:
            if dep in completed_ids:
                # In a real system, you'd look up the result
                pass
        
        # Execute the task
        result = execute_task(task, {})
        task.result = result
        task.status = TaskStatus.COMPLETED
        
        completed_ids.add(task.id)
        print(f"✅ {task.id} complete")
    
    return {
        "tasks": tasks,
        "completed_task_ids": list(completed_ids),
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [AIMessage(content=f"Executed {len(ready)} tasks")]
    }


def route_after_llm(state: AgentState):
    """Route after LLM to check for HITL needs."""
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_msg = messages[-1]
    
    # Check if the AI made tool calls
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        high_stakes = {"send_email", "post_slack", "save_file", "finalize_report"}
        for tc in last_msg.tool_calls:
            if tc.get("name") in high_stakes:
                return "hitl_gate"
        return "tools"
    
    # Check if we have tasks to execute
    tasks = state.get("tasks", [])
    pending = [t for t in tasks if t.status != TaskStatus.COMPLETED]
    if pending:
        return "executor"
    
    return END


def build_complete_graph():
    """Build the complete Day 3 agent graph."""
    conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
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
    
    # Conditional routing
    builder.add_conditional_edges("llm", route_after_llm, {
        "executor": "executor",
        "tools": "tools",
        "hitl_gate": "hitl_gate",
        END: END
    })
    
    builder.add_edge("hitl_gate", "tools")
    builder.add_edge("tools", "llm")
    
    return builder.compile(checkpointer=memory)


def llm_node(state: AgentState):
    """LLM reasoning node."""
    # In production, this would call the LLM with tools
    return {"messages": [AIMessage(content="LLM processed the request.")]}


def tool_node(state: AgentState):
    """Tool execution node."""
    # In production, this would execute tools
    return {"messages": [AIMessage(content="Tools executed.")]}


if __name__ == "__main__":
    graph = build_complete_graph()
    
    config = {"configurable": {"thread_id": "day3_final"}}
    
    # Run the agent
    goal = "Research competitor Tesla and produce a battlecard"
    result = graph.invoke(
        {"messages": [HumanMessage(content=goal)]},
        config=config
    )
    
    print("\n=== FINAL STATE ===")
    print(f"Messages: {len(result.get('messages', []))}")
    print(f"Tasks: {len(result.get('tasks', []))}")
    print(f"Completed: {len(result.get('completed_task_ids', []))}")