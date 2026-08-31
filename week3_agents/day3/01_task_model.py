"""
01_task_models.py
Defines the Task data model and TaskStatus enum.
These are the building blocks for planning and execution.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class TaskStatus(str, Enum):
    """Status of a task in the execution pipeline."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """A single atomic unit of work in the task tree."""
    
    id: str = Field(description="Unique task identifier, e.g., 'T1', 'T2'")
    description: str = Field(description="What this task does")
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one starts"
    )
    assigned_tool: Optional[str] = Field(
        default=None,
        description="Which tool to use: 'web_search', 'calculator', 'analyst'"
    )
    result: Optional[str] = Field(
        default=None,
        description="Output of this task after execution"
    )
    
    def is_ready(self, completed_ids: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.dependencies)