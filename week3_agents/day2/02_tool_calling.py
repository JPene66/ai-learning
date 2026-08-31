"""
MULTI-TOOL REACT AGENT — Complete Working Example
Run this file to see the agent with 3 tools in action.
"""

# ============================================================
# STEP 1: IMPORTS
# ============================================================
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from ddgs import DDGS
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# STEP 2: DEFINE THE AGENT STATE
# ============================================================
class AgentState(TypedDict):
    # The messages list grows with every turn.
    # add_messages ensures new messages are appended, not overwritten.
    messages: Annotated[list, add_messages]


# ============================================================
# STEP 3: DEFINE THE CALCULATOR TOOL
# ============================================================
def safe_calculator_expression(expression: str) -> str:
    """Safely evaluate a math expression."""
    # Block dangerous operations
    dangerous = ['import', 'os', 'sys', 'subprocess', 'open', '__', 'exec', 'eval']
    if any(d in expression.lower() for d in dangerous):
        return "Error: Expression contains unsafe operations."

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A valid Python math expression as a string.
                   Examples: '2 + 2', '15 * 4', '100 / 3'
    """
    return safe_calculator_expression(expression)


# ============================================================
# STEP 4: DEFINE THE WEB SEARCH TOOL
# ============================================================
@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: A specific, concise search query. 
               Good: 'Tesla Q3 2024 revenue'
               Bad: 'Tell me about Tesla'
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            if not results:
                return "No results found for this query."

            # Format results into a readable string
            output = []
            for i, r in enumerate(results, 1):
                output.append(f"{i}. {r['title']}: {r['body'][:200]}...")
            return "\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ============================================================
# STEP 5: DEFINE THE WEATHER TOOL (SIMULATED)
# ============================================================
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city. Use the English name.
              Examples: 'London', 'New York', 'Tokyo'
    """
    # Simulated response for learning purposes
    weather_db = {
        "london": "15°C, cloudy with occasional rain",
        "new york": "22°C, sunny",
        "tokyo": "28°C, humid with thunderstorms",
        "paris": "18°C, partly cloudy",
        "lagos": "30°C, sunny with high humidity",
        "accra": "28°C, partly cloudy"
    }

    normalized = city.lower().strip()
    if normalized in weather_db:
        return f"Weather in {city}: {weather_db[normalized]}"

    return f"Weather data for '{city}' is not available in our database."


# ============================================================
# STEP 6: SETUP THE LLM WITH TOOLS
# ============================================================
# All available tools
tools = [calculator, web_search, get_weather]

# Initialize model with tools bound
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful assistant with access to three tools:
1. calculator - for ALL math calculations
2. web_search - MANDATORY for answering questions about facts, history, current events, and general knowledge. Even if you think you know the answer, you MUST use web_search to verify it.
3. get_weather - for weather information in specific cities

Use the right tool for each question. Always explain your reasoning first."""


# ============================================================
# STEP 7: DEFINE THE LLM NODE (THINK)
# ============================================================
def llm_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]+state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ============================================================
# STEP 8: DEFINE THE TOOL NODE (ACT)
# ============================================================
tool_node = ToolNode(tools)


# ============================================================
# STEP 9: BUILD THE GRAPH (WIRE THE LOOP)
# ============================================================
builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.set_entry_point("llm")

def route_after_llm(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")

multi_tool_agent = builder.compile()


# ============================================================
# STEP 10: RUN THE AGENT WITH DIFFERENT QUESTIONS
# ============================================================
def run_agent(question: str, agent=multi_tool_agent):
    """Run the agent and print the conversation with proper formatting."""
    print("\n" + "=" * 70)
    print(f" QUESTION: {question}")
    print("=" * 70)

    # Run the graph
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    # Print the full conversation trace with better formatting
    tool_calls_made = []
    
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.type.upper()
        
        if msg_type == "HUMAN":
            print(f"\n👤 USER: {msg.content}")
            
        elif msg_type == "AI":
            # Check if this message has tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\n AGENT REASONING: {msg.content}")
                print(f"    TOOL CALLS:")
                for tc in msg.tool_calls:
                    print(f"      → {tc['name']}({tc['args']})")
                    tool_calls_made.append(tc['name'])
            else:
                print(f"\n AGENT FINAL ANSWER: {msg.content}")
                
        elif msg_type == "TOOL":
            print(f"\n TOOL RESULT: {msg.content[:200]}...")
            if len(msg.content) > 200:
                print(f"   (truncated, full response length: {len(msg.content)} characters)")

    print("\n" + "=" * 70)
    print(f" TOOLS USED: {', '.join(set(tool_calls_made)) if tool_calls_made else 'None'}")
    print("=" * 70)


# ============================================================
# STEP 11: TEST WITH DIFFERENT QUESTIONS
# ============================================================
if __name__ == "__main__":
    print("\n" + "*" * 35)
    print("MULTI-TOOL REACT AGENT — DEMO")
    print("*" * 35)
    
    # Test 1: Math → should use calculator
    run_agent("What is 144 divided by 12?")
    
    # Test 2: Weather → should use get_weather
    run_agent("What's the weather like in Tokyo?")
    
    # Test 3: Current info → should use web_search
    run_agent("Who won the FIFA World Cup in 2022?")
    
    # Test 4: General knowledge → should use web_search
    run_agent("What is the capital of Australia?")
    
    # Test 5: Complex question requiring reasoning
    run_agent("If the temperature in Paris is 18°C, and I want to visit New York where it's 22°C, what's the temperature difference?")
    
    print("\n" + "#" * 35)
    print("DEMO COMPLETE — All 5 tests run successfully!")
    print("#" * 35)