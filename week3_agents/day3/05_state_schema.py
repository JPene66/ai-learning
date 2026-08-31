"""
05_state_schema.py
Defines the AgentState with custom reducers for merging state updates.
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages
from operator import add

from 01_task_models import Task


# Custom reducer: merge task lists instead of replacing
def merge_tasks(existing: List[Task], new_tasks: List[Task]) -> List[Task]:
    """Merge task lists, updating existing tasks by id."""
    task_map = {t.id: t for t in existing}
    for t in new_tasks:
        task_map[t.id] = t
    return list(task_map.values())


class AgentState(TypedDict):
    """The complete state of the agent."""
    
    # Conversation history (appends new messages)
    messages: Annotated[list, add_messages]
    
    # Current task plan (updates by ID)
    tasks: Annotated[List[Task], merge_tasks]
    
    # Which task is running now
    current_task_id: Optional[str]
    
    # Accumulates completed task IDs
    completed_task_ids: Annotated[List[str], add]
    
    # Loop guard counter
    iteration_count: int
    
    # Running cost tracker
    total_cost: float
    
    # For resume after crash
    checkpoint_id: Optional[str]