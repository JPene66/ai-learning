"""
REACT AGENT - Complete Working Example
"""

# Import Libraries
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv

load_dotenv()

# === DEFINE THE AGENT STATE ===

class AgentState(TypedDict):
    # The message list grows with every turn
    # add_messages ensure new messages are appended to the old ones
    messages: Annotated[list, add_messages]

# === DEFINE CALCULATOR TOOL ===
@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.

    Args:
        expression (str): The mathematical expression to evaluate (e.g., "2+2").

    Returns:
        str: The result of the expression as a string.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# === SETUP THE LLM ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===  LIST ALL THE TOOL THE AGENT CAN USE ===
tools = [calculator]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """ You are a helpful assistant that solves problems step by step. When you need to perform a calculation, you MUST use the calculator tool. Always explain your reasoning before taking any action. """

def llm_node(state: AgentState):
    # Append the system prompt to the conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    # LLM processes the messages
    response = llm_with_tools.invoke(messages)
    # Return the response
    return {"messages": [response]}

# === TOOLNODE IS A LANGGRAPH BUILT IN THAT HANDLES TOOLS EXECUTION
tool_node = ToolNode(tools)

# BUILD THE GRAPH (WIRE THE LOOP)
builder = StateGraph(AgentState)

# ADD NOTES
builder.add_node("llm", llm_node) # The thinking node
builder.add_node("tools", tool_node) # The acting node

builder.set_entry_point("llm")
# CONDITIONAL EDGE: DECIDE WHERE TO GO NEXT (LLM -> TOOLS or END)
def should_continue(state:AgentState):
    last_message = state["messages"][-1]

    # If the last message has tool_calls, then go to tool_node
    if last_message.tool_calls:
        return 'tools'
    
    else:
        return END

# ADD THE CONDITIONAL EDGE FROM LLM
builder.add_conditional_edges("llm", should_continue, {"tools": "tools", END:END})

builder.add_edge("tools", "llm")

# COMPILE THE GRAPH
graph = builder.compile()

def run_agent(question: str):
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    for msg in result["messages"]:
        print(f"\n{msg.type.upper()}: {msg.content}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\nTOOL CALL: {tc['name']}({tc['args']})")

    print("\n" + "=" * 60)
    print(f"FINAL ANSWER: {result['messages'][-1].content}")
    print("\n" + "=" * 60)


# TEST IT
if __name__ == "__main__":
    run_agent("What is the population of the capital of France divided by 4")
    
        
    
