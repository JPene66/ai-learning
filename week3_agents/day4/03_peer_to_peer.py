"""
03_peer_to_peer.py
Implements the peer-to-peer debate loop with Analyst and Critic.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from 01_agent_roles import ANALYST_PROMPT, CRITIC_PROMPT
from 01_agent_roles import get_analyst_llm, get_critic_llm


class DebateState(TypedDict):
    """State for the debate loop."""
    messages: Annotated[list, add_messages]
    research_findings: str
    analysis_output: str
    critique_output: str
    debate_round: int           # Track how many back-and-forths
    max_debate_rounds: int      # Safety limit
    critic_satisfied: bool      # Loop exit condition
    final_report: str


# ============================================================
# DEBATE CRITIC NODE
# ============================================================
def debate_critic_node(state: DebateState):
    """Critic reviews and decides if more revision is needed."""
    
    analysis = state["analysis_output"]
    findings = state["research_findings"]
    round_num = state.get("debate_round", 1)
    max_rounds = state.get("max_debate_rounds", 3)
    
    prompt = f"""{CRITIC_PROMPT}

This is debate round {round_num} of {max_rounds}.

Analyst's current battlecard:
{analysis}

Original findings:
{findings}

Provide your critique. At the end, explicitly state:
SATISFIED: YES or NO
(If NO, explain what must be fixed for you to approve)"""
    
    response = get_critic_llm().invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Parse satisfaction from response
    satisfied = "SATISFIED: YES" in content.upper() or round_num >= max_rounds
    
    return {
        "critique_output": content,
        "critic_satisfied": satisfied,
        "debate_round": round_num + 1,
        "messages": [AIMessage(content=f"[Critic Round {round_num}] {content}")]
    }


# ============================================================
# DEBATE ANALYST NODE (REVISION)
# ============================================================
def debate_analyst_node(state: DebateState):
    """Analyst revises based on Critic's feedback."""
    
    prompt = f"""{ANALYST_PROMPT}

Your previous battlecard was critiqued. Here is the critique:
{state['critique_output']}

Revise your battlecard to address ALL issues raised. Maintain source attribution.
Return the COMPLETE revised battlecard (not just changes)."""
    
    response = get_analyst_llm().invoke([HumanMessage(content=prompt)])
    
    return {
        "analysis_output": response.content,
        "messages": [AIMessage(content=f"[Analyst Revision] {response.content}")]
    }


# ============================================================
# ROUTING FOR DEBATE LOOP
# ============================================================
def route_debate(state: DebateState):
    """Route to finalize if satisfied, otherwise loop back to revision."""
    if state.get("critic_satisfied", False):
        return "finalize"
    return "revise"  # Go back to analyst for revision


# ============================================================
# FINALIZER FOR DEBATE
# ============================================================
def debate_finalizer(state: DebateState):
    """Synthesize the final report after debate is complete."""
    
    synthesis_prompt = f"""You are the Finalizer. Combine the Analyst's battlecard with the Critic's feedback.

BATTLECARD:
{state['analysis_output']}

CRITIQUE:
{state['critique_output']}

Instructions:
- Incorporate valid critiques into the final report
- Add a "Revision Notes" section explaining what changed
- Maintain source attribution for all claims
- Format as a professional markdown report"""
    
    response = get_finalizer_llm().invoke([HumanMessage(content=synthesis_prompt)])
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"[Final Report]\n{response.content}")]
    }


# ============================================================
# BUILD THE DEBATE GRAPH
# ============================================================
def build_debate_graph():
    """Build and compile the debate graph."""
    
    builder = StateGraph(DebateState)
    
    # Add nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", debate_analyst_node)
    builder.add_node("critic", debate_critic_node)
    builder.add_node("finalizer", debate_finalizer)
    
    # Set entry point
    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "analyst")
    builder.add_edge("analyst", "critic")
    
    # Conditional routing after critic
    builder.add_conditional_edges("critic", route_debate, {
        "revise": "analyst",   # Loop back to analyst
        "finalize": "finalizer"
    })
    
    builder.add_edge("finalizer", END)
    
    return builder.compile()


# ============================================================
# RESEARCHER NODE FOR DEBATE
# ============================================================
def researcher_node(state: DebateState):
    """Placeholder researcher node for the debate graph."""
    # In production, this would do actual research
    findings = "Tesla Q3 2024 revenue: $25.18B. Employee count: 140,000. Key products: Model Y, Model 3."
    return {
        "research_findings": findings,
        "messages": [AIMessage(content=f"[Researcher] {findings}")]
    }


# ============================================================
# GET FINALIZER LLM
# ============================================================
def get_finalizer_llm():
    """Get the finalizer LLM."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ============================================================
# RUN THE DEBATE
# ============================================================
if __name__ == "__main__":
    graph = build_debate_graph()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="Research Tesla and analyze")],
        "debate_round": 1,
        "max_debate_rounds": 3,
        "critic_satisfied": False
    })
    
    print("\n" + "=" * 60)
    print("FINAL REPORT AFTER DEBATE")
    print("=" * 60)
    print(result["final_report"])
    
    print(f"\n📊 Debate Rounds: {result.get('debate_round', 0) - 1}")
    print(f"✅ Critic Satisfied: {result.get('critic_satisfied', False)}")