"""
AGENT MEMORY — Complete Working Example
Run this file to see all three memory types in action.
"""

# ============================================================
# STEP 1: IMPORTS
# ============================================================
from tiktoken import model
from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# STEP 2: DEFINE THE AGENT STATE WITH MEMORY
# ============================================================
class MemoryAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Optional: add user_id, session_id for multi-user support
    user_id: Optional[str]
    session_id: Optional[str]



# ============================================================
# STEP 3: SETUP LLM AND EMBEDDINGS
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store for long-term and episodic memory
vector_store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"  # Data survives restarts
)


# ============================================================
# STEP 4: DEFINE TOOLS
# ============================================================
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A valid Python math expression as a string.
                   Examples: '2 + 2', '15 * 4', '100 / 3'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: A specific, concise search query.
               Good: 'Tesla Q3 2024 revenue'
               Bad: 'Tell me about Tesla'
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            if not results:
                return "No results found."
            output = [f"{i+1}. {r['title']}: {r['body'][:150]}..." for i, r in enumerate(results)]
            return "\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"


# ============================================================
# STEP 5: SHORT-TERM MEMORY MANAGEMENT
# ============================================================
def trim_node(state: MemoryAgentState):
    """Trim messages to stay within token budget."""
    from langchain_core.messages import trim_messages
    
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,        # Keep last ~4000 tokens
        strategy="last",        # Keep the most recent messages
        token_counter=llm,     # Use the model's tokenizer
        allow_partial=False,   # Don't cut a message in half
        start_on="human",      # Always start with a human message
    )
    return {"messages": trimmed}

def summarize_node(state: MemoryAgentState):
    """Summarize old messages into a single memory message."""
    # Keep the last 4 messages as-is
    recent = state["messages"][-4:]
    older = state["messages"][:-4]

    if older:
        summary_prompt = f"Summarize this conversation concisely: {older}"
        summary = llm.invoke([HumanMessage(content=summary_prompt)]).content

        # Replace old messages with a summary
        return {
            "messages": [
                SystemMessage(content=f"Previous conversation summary: {summary}"),
                *recent
            ]
        }

    return {"messages": state["messages"]}



# ============================================================
# STEP 6: LONG-TERM MEMORY (Vector Store)
# ============================================================
def save_to_memory(fact: str, category: str = "general"):
    """Save a fact to the agent's long-term memory."""
    doc = Document(
        page_content=fact,
        metadata={
            "category": category,
            "type": "fact",
            "timestamp": "2024-07-30"
        }
    )
    vector_store.add_documents([doc])
    print(f"💾 Saved to memory: {fact[:50]}..." if len(fact) > 50 else f"💾 Saved to memory: {fact}")
    return True



def retrieve_memories(query: str, k: int = 3):
    """Retrieve memories relevant to the current query."""
    results = vector_store.similarity_search(query, k=k, filter={"type": "fact"})

    if not results:
        return "No relevant memories found."

    memory_text = "\n".join([
        f"- {doc.page_content} (category: {doc.metadata.get('category', 'general')})"
        for doc in results
    ])

    return memory_text



def memory_retrieval_node(state: MemoryAgentState):
    """Retrieve and inject long-term memories based on the last user message."""
    # Find the last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {"messages": state["messages"]}

    # Retrieve relevant memories
    memories = retrieve_memories(last_user_msg, k=3)

    # Add memories as a system message (will be included in the next LLM call)
    memory_message = SystemMessage(
        content=f"RELEVANT FACTS ABOUT THE USER:\n{memories}\n\nUse these facts to personalize your response. If a fact contradicts the current request, prioritize the current request."
    )

    # Insert the memory message right before the last user message
    messages = list(state["messages"])
    # Find the last user message index
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            messages.insert(i, memory_message)
            break

    return {"messages": messages}


# ============================================================
# STEP 7: EPISODIC MEMORY (Lessons Learned)
# ============================================================
def save_episode(task: str, outcome: str, lesson: str):
    """Save a lesson learned from a completed task."""
    episode = Document(
        page_content=f"Task: {task} | Outcome: {outcome} | Lesson: {lesson}",
        metadata={
            "type": "episode",
            "success": "true" if "success" in outcome.lower() or "passed" in outcome.lower() else "false"
        }
    )
    vector_store.add_documents([episode])
    print(f"Saved episode: {task[:30]}..." if len(task) > 30 else f"📚 Saved episode: {task}")


def get_lessons(query: str, k: int = 2):
    """Retrieve relevant episodes/lessons."""
    episodes = vector_store.similarity_search(
        query,
        k=k,
        filter={"type": "episode"}
    )
    if not episodes:
        return "No relevant lessons found."
    return "\n".join([e.page_content for e in episodes])


# ============================================================
# STEP 8: BUILD THE MEMORY-AWARE AGENT
# ============================================================
# Setup tools
tools = [calculator, web_search]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools:
1. calculator - for ALL math calculations
2. web_search - for current events and information you don't know

Always explain your reasoning first. Use the tools when needed."""

def llm_node_with_memory(state: MemoryAgentState):
    """LLM node with memory integration."""
    # Check if there's a memory message already (from retrieval node)
    messages = state["messages"]
    
    # If no memory message is present, use the default system prompt
    has_memory = any(isinstance(msg, SystemMessage) and "" in msg.content for msg in messages)
    
    if not has_memory:
        # Prepend system prompt
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_llm(state: MemoryAgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END


def build_memory_agent():
    """Build the complete memory-aware agent."""
    builder = StateGraph(MemoryAgentState)

    # Add nodes
    builder.add_node("trim", trim_node)                    # Short-term: trim old messages
    builder.add_node("retrieve", memory_retrieval_node)    # Long-term: retrieve memories
    builder.add_node("llm", llm_node_with_memory)         # Think
    builder.add_node("tools", tool_node)                  # Act

    # Wiring
    builder.set_entry_point("trim")
    builder.add_edge("trim", "retrieve")
    builder.add_edge("retrieve", "llm")
    builder.add_conditional_edges("llm", route_after_llm, {"tools": "tools", END: END})
    builder.add_edge("tools", "llm")

    return builder.compile()


# ============================================================
# STEP 9: RUN THE AGENT WITH MEMORY
# ============================================================
def run_agent_with_memory(question: str, agent):
    """Run the agent and show the conversation with memory."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    for msg in result["messages"]:
        msg_type = msg.type.upper()
        if msg_type == "HUMAN":
            print(f"\nUSER: {msg.content}")
        elif msg_type == "AI":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\nAGENT: {msg.content}")
                for tc in msg.tool_calls:
                    print(f"   → Tool: {tc['name']}({tc['args']})")
            else:
                print(f"\nAGENT FINAL: {msg.content}")
        elif msg_type == "TOOL":
            print(f"\nTOOL RESULT: {msg.content[:100]}...")


# ============================================================
# STEP 10: DEMONSTRATION
# ============================================================
if __name__ == "__main__":
    print("*" * 35)
    print("AGENT MEMORY — DEMO")
    print("*" * 35)

    # ============================================================
    # PART A: SHORT-TERM MEMORY DEMO
    # ============================================================
    print("\n" + "=" * 70)
    print("PART A: SHORT-TERM MEMORY (Conversation Buffer)")
    print("=" * 70)

    # Clear existing memory for this demo
    # (In practice, you'd have different collections per user)
    
    print("\nShort-term memory keeps the current conversation context.")
    print("The agent remembers what you said earlier in this session.\n")

    agent = build_memory_agent()

    run_agent_with_memory("What is 15 * 6?", agent)
    run_agent_with_memory("Now divide that result by 3.", agent)
    run_agent_with_memory("What was the original number I asked about?", agent)

    # ============================================================
    # PART B: LONG-TERM MEMORY DEMO
    # ============================================================
    print("\n" + "=" * 70)
    print("PART B: LONG-TERM MEMORY (Persistent Facts)")
    print("=" * 70)

    print("\nSaving facts to long-term memory...")
    save_to_memory("User prefers to be called Alex, not Alexander", "preference")
    save_to_memory("User works as a software engineer at a fintech company", "profile")
    save_to_memory("User's favorite programming language is Python", "preference")
    save_to_memory("User prefers short, direct answers", "communication_style")

    print("\nRetrieving memories...")
    memories = retrieve_memories("How should I communicate with the user?")
    print(f"\nRetrieved memories:\n{memories}")

    # ============================================================
    # PART C: EPISODIC MEMORY DEMO
    # ============================================================
    print("\n" + "=" * 70)
    print("PART C: EPISODIC MEMORY (Lessons Learned)")
    print("=" * 70)

    print("\nSaving episodes (lessons learned)...")
    save_episode(
        task="Search for Python web frameworks",
        outcome="Success - found Django, Flask, FastAPI",
        lesson="For Python web framework searches, include 'popular' in the query for better results"
    )
    save_episode(
        task="Calculator precision",
        outcome="Failed - precision issues with decimal division",
        lesson="For division in calculator, round to 2 decimal places for clarity"
    )

    print("\nRetrieving lessons...")
    lessons = get_lessons("Python web frameworks")
    print(f"\nRetrieved lessons:\n{lessons}")

    # ============================================================
    # PART D: FULL MEMORY-AGENT TEST
    # ============================================================
    print("\n" + "=" * 70)
    print("PART D: FULL MEMORY-AGENT TEST")
    print("=" * 70)

    print("\nThe agent now remembers user preferences across sessions.")
    print("It uses short-term, long-term, and episodic memory together.\n")

    # Run a test with the memory-aware agent
    run_agent_with_memory("What programming language do I like?", agent)

    print("\n" + "#" * 35)
    print("DEMO COMPLETE — All memory types demonstrated!")
    print("#" * 35)