"""
04_replanning.py
Adds automatic replanning when a task fails.
"""

import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from 01_task_models import Task, TaskStatus
from 02_planner import llm, planner_node
from 03_executor import execute_task

planning_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def executor_with_replanning(tasks: List[Task], goal: str) -> List[Task]:
    """Execute with automatic replanning on failure."""
    completed_ids = set()
    completed_results = {}
    pending = tasks.copy()
    max_iterations = 50
    iteration = 0
    
    while pending and iteration < max_iterations:
        iteration += 1
        ready = [t for t in pending if t.is_ready(completed_ids)]
        
        if not ready:
            break
        
        for task in ready:
            task.status = TaskStatus.IN_PROGRESS
            result = execute_task(task, completed_results)
            
            # Check for failure
            if "Error" in result or "Failed" in result:
                task.status = TaskStatus.FAILED
                print(f"❌ {task.id} FAILED: {result}")
                
                # REPLAN: Ask the LLM for a recovery plan
                replan_prompt = f"""Task {task.id} failed: {task.description}
Failure reason: {result}
Current completed tasks: {completed_results}
Remaining tasks: {[t.id for t in pending]}

Generate a NEW task (or tasks) to recover from this failure.
Output a single JSON task object."""
                
                recovery_response = planning_llm.invoke([HumanMessage(content=replan_prompt)])
                try:
                    recovery_task = Task(**json.loads(recovery_response.content))
                    recovery_task.dependencies = list(completed_ids)  # Depends on everything so far
                    pending.append(recovery_task)
                    print(f"🔄 Replanning: added recovery task {recovery_task.id}")
                except:
                    print("Replanning failed. Escalating to human.")
                    # In production: trigger HITL here
                
                pending.remove(task)
            else:
                task.result = result
                task.status = TaskStatus.COMPLETED
                completed_ids.add(task.id)
                completed_results[task.id] = result
                pending.remove(task)
    
    return tasks


# Test replanning
if __name__ == "__main__":
    goal = "Research competitor Tesla and produce a battlecard"
    tasks = planner_node(goal)
    completed_tasks = executor_with_replanning(tasks, goal)
    
    for t in completed_tasks:
        print(f"{t.id}: {t.status} - {t.result[:50] if t.result else 'No result'}...")