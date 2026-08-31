"""
02_planner.py
The planner uses an LLM to break a high-level goal into a structured task tree.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from typing import List
from dotenv import load_dotenv

from 01_task_model import Task

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PLANNER_PROMPT = """You are a task planner. Break the user's goal into a JSON list of tasks.

Rules:
1. Each task must have a unique id (T1, T2, ...)
2. Use dependencies to enforce order. If Task B needs Task A's output, set B.dependencies = ["T1"]
3. Assign the most appropriate tool to each task from: web_search, analyst, calculator
4. Keep tasks atomic — one clear action per task
5. Output ONLY valid JSON. No markdown, no explanations.

Example output:
[
{
"id": "T1",
"description": "Search for Tesla's Q3 2024 revenue",
"status": "pending",
"dependencies": [],
"assigned_tool": "web_search",
"result": null
},
{
"id": "T2",
"description": "Search for Tesla's employee count",
"status": "pending",
"dependencies": [],
"assigned_tool": "web_search",
"result": null
},
{
"id": "T3",
"description": "Analyze Tesla's financial health from revenue and employee data",
"status": "pending",
"dependencies": ["T1", "T2"],
"assigned_tool": "analyst",
"result": null
}
]"""


def planner_node(goal: str) -> List[Task]:
    """Decompose a goal into a task tree using the LLM."""
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Goal: {goal}")
    ]
    
    response = llm.invoke(messages)
    
    try:
        task_dicts = json.loads(response.content)
        tasks = [Task(**t) for t in task_dicts]
        return tasks
    except json.JSONDecodeError as e:
        print(f"Planner returned invalid JSON: {e}")
        print(f"Raw output: {response.content}")
        return []


# Test the planner
if __name__ == "__main__":
    goal = "Research competitor Tesla and produce a battlecard with strengths, weaknesses, and pricing"
    tasks = planner_node(goal)
    for t in tasks:
        print(f"{t.id}: {t.description} [deps: {t.dependencies}] [tool: {t.assigned_tool}]")