"""
07_state_inspection.py
Utilities for inspecting and debugging agent state.
"""

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Reconnect to the same database
conn = sqlite3.connect("agent_checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)


def inspect_state(thread_id: str):
    """Inspect the full state snapshot for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = memory.get(config)
    
    if not state:
        print(f"No state found for thread: {thread_id}")
        return
    
    print(f"\n=== STATE INSPECTION: {thread_id} ===")
    print(f"Messages: {len(state.get('messages', []))}")
    print(f"Tasks: {len(state.get('tasks', []))}")
    print(f"Current task: {state.get('current_task_id')}")
    print(f"Completed: {state.get('completed_task_ids', [])}")
    print(f"Iterations: {state.get('iteration_count', 0)}")
    print(f"Cost so far: ${state.get('total_cost', 0):.4f}")
    
    # List all checkpoints
    print(f"\n--- Checkpoints ---")
    for checkpoint in memory.list(config):
        print(f"Step {checkpoint['step']}: {checkpoint['ts']}")


def list_all_threads():
    """List all thread IDs with checkpoints."""
    # Query the SQLite database directly
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
    threads = cursor.fetchall()
    
    print("\n=== ALL THREADS ===")
    for thread_id in threads:
        print(f"- {thread_id[0]}")


if __name__ == "__main__":
    list_all_threads()
    inspect_state("session_001")