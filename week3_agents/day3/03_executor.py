"""
03_executor.py
Executes tasks in the correct order, respecting dependencies.
"""

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from 01_task_model import Task, TaskStatus
from 02_planner import planner_node

analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


def execute_task(task: Task, completed_results: Dict[str, str]) -> str:
    """Execute a single task using its assigned tool."""
    
    # Inject dependency results into the task description
    context = "\n".join([
        f"Result from {dep_id}: {completed_results.get(dep_id, 'N/A')}"
        for dep_id in task.dependencies
    ])
    
    full_prompt = f"Task: {task.description}\n\nContext from previous tasks:\n{context}"
    
    if task.assigned_tool == "web_search":
        return f"[Simulated search result for: {task.description}]"
    
    elif task.assigned_tool == "analyst":
        response = analyst_llm.invoke([HumanMessage(content=full_prompt)])
        return response.content
    
    elif task.assigned_tool == "calculator":
        return f"[Calculated: {task.description}]"
    
    else:
        return f"Unknown tool: {task.assigned_tool}"


def executor(tasks: List[Task]) -> List[Task]:
    """Execute all tasks in dependency order."""
    completed_ids = set()
    completed_results = {}
    pending = tasks.copy()
    max_iterations = 50  # Safety guard
    iteration = 0
    
    while pending and iteration < max_iterations:
        iteration += 1
        
        # Find tasks that are ready to run
        ready = [t for t in pending if t.is_ready(completed_ids)]
        
        if not ready:
            print("ERROR: No tasks ready but tasks remain. Check dependencies.")
            break
        
        for task in ready:
            print(f"\n▶️ Executing {task.id}: {task.description}")
            task.status = TaskStatus.IN_PROGRESS
            
            result = execute_task(task, completed_results)
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            completed_ids.add(task.id)
            completed_results[task.id] = result
            pending.remove(task)
            
            print(f"✅ {task.id} complete: {result[:100]}...")
    
    return tasks


# Test the full pipeline
if __name__ == "__main__":
    goal = "Research competitor Tesla and produce a battlecard with strengths, weaknesses, and pricing"
    tasks = planner_node(goal)
    completed_tasks = executor(tasks)
    
    print("\n=== FINAL BATTLECARD ===")
    for t in completed_tasks:
        if t.assigned_tool == "analyst":
            print(f"\n{t.description}:\n{t.result}")