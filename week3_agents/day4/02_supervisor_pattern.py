"""
02_supervisor_pattern.py
Implements the Supervisor pattern with sequential routing.
"""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from 01_agent_roles import (
    RESEARCHER_PROMPT, ANALYST_PROMPT, CRITIC_PROMPT, FINALIZER_PROMPT,
    get_researcher_llm, get_analyst_llm, get_critic_llm, get_finalizer_llm
)


class MultiAgentState(TypedDict):
    """State for the multi-agent system."""
    messages: Annotated[list, add_messages]
    research_findings: str      # Output from Researcher
    analysis_output: str        # Output from Analyst
    critique_output: str        # Output from Critic
    final_report: str           # Synthesized output
    current_step: str           # Tracks which agent is running


# ============================================================
# SUPERVISOR NODE
# ============================================================
def supervisor_node(state: MultiAgentState):
    """The supervisor decides what to do next based on current progress."""
    
    step = state.get("current_step", "start")
    
    if step == "start":
        return {"current_step": "research"}
    
    elif step == "research":
        return {"current_step": "analysis"}
    
    elif step == "analysis":
        return {"current_step": "critique"}
    
    elif step == "critique":
        return {"current_step": "finalize"}
    
    else:
        return {"current_step": END}


# ============================================================
# RESEARCHER NODE
# ============================================================
def researcher_node(state: MultiAgentState):
    """The Researcher fetches raw facts about the competitor."""
    
    # Get the user's original question
    user_msg = state["messages"][0].content
    
    prompt = f"""{RESEARCHER_PROMPT}

Your task: Research facts about the competitor mentioned in this request.
Request: {user_msg}

Search for: revenue, employee count, key products, recent news, and partnerships.
Return structured findings with sources."""
    
    response = get_researcher_llm().invoke([HumanMessage(content=prompt)])
    
    return {
        "research_findings": response.content,
        "messages": [AIMessage(content=f"[Researcher] {response.content}")]
    }


# ============================================================
# ANALYST NODE
# ============================================================
def analyst_node(state: MultiAgentState):
    """The Analyst synthesizes findings into a battlecard."""
    
    findings = state["research_findings"]
    
    prompt = f"""{ANALYST_PROMPT}

Here are the Researcher's findings:
{findings}

Synthesize these into a structured battlecard with:
1. Executive Summary (2 sentences)
2. Strengths (bullet points with evidence)
3. Weaknesses (bullet points with evidence)
4. Opportunities (bullet points with evidence)
5. Threats (bullet points with evidence)"""
    
    response = get_analyst_llm().invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis_output": response.content,
        "messages": [AIMessage(content=f"[Analyst] {response.content}")]
    }


# ============================================================
# CRITIC NODE
# ============================================================
def critic_node(state: MultiAgentState):
    """The Critic reviews the Analyst's battlecard for issues."""
    
    analysis = state["analysis_output"]
    findings = state["research_findings"]
    
    prompt = f"""{CRITIC_PROMPT}

Analyst's battlecard:
{analysis}

Original research findings:
{findings}

Provide your critique. Identify at least 3 specific issues."""
    
    response = get_critic_llm().invoke([HumanMessage(content=prompt)])
    
    return {
        "critique_output": response.content,
        "messages": [AIMessage(content=f"[Critic] {response.content}")]
    }


# ============================================================
# FINALIZER NODE
# ============================================================
def finalizer_node(state: MultiAgentState):
    """Synthesize everything into the final report."""
    
    synthesis_prompt = f"""{FINALIZER_PROMPT}

BATTLECARD:
{state['analysis_output']}

CRITIQUE:
{state['critique_output']}

Instructions:
- Incorporate valid critiques into the final report
- Add a "Revision Notes" section explaining what changed based on feedback
- Maintain source attribution for all claims
- Format as a professional markdown report"""
    
    response = get_finalizer_llm().invoke([HumanMessage(content=synthesis_prompt)])
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"[Final Report]\n{response.content}")],
        "current_step": END
    }


# ============================================================
# BUILD THE SUPERVISOR GRAPH
# ============================================================
def build_supervisor_graph():
    """Build and compile the supervisor graph."""
    
    builder = StateGraph(MultiAgentState)
    
    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)
    builder.add_node("finalizer", finalizer_node)
    
    # The supervisor routes to the appropriate worker
    def route_from_supervisor(state: MultiAgentState):
        return state["current_step"]
    
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", route_from_supervisor, {
        "research": "researcher",
        "analysis": "analyst",
        "critique": "critic",
        "finalize": "finalizer",
        END: END
    })
    
    # After each worker, return to supervisor for next routing
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("critic", "supervisor")
    builder.add_edge("finalizer", END)
    
    return builder.compile()


# ============================================================
# RUN THE SUPERVISOR
# ============================================================
if __name__ == "__main__":
    graph = build_supervisor_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="Research competitor Tesla")],
        "current_step": "start"
    })
    
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])