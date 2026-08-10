"""
08_multi_agent_system.py
Complete 3-agent competitive intelligence system.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from 01_agent_roles import (
    RESEARCHER_PROMPT, ANALYST_PROMPT, CRITIC_PROMPT,
    get_researcher_llm, get_analyst_llm, get_critic_llm
)
from 02_supervisor_pattern import (
    MultiAgentState, supervisor_node, researcher_node,
    analyst_node, critic_node, finalizer_node
)
from 03_peer_to_peer import (
    DebateState, debate_critic_node, debate_analyst_node,
    route_debate, get_finalizer_llm
)
from 05_blackboard import blackboard, BlackboardEntry
from 04_message_protocol import AgentMessage, MessageType


# ============================================================
# COMBINED SYSTEM STATE
# ============================================================
class CombinedState(TypedDict):
    messages: Annotated[list, add_messages]
    research_findings: str
    analysis_output: str
    critique_output: str
    final_report: str
    current_step: str
    debate_round: int
    max_debate_rounds: int
    critic_satisfied: bool


# ============================================================
# COMBINED NODES
# ============================================================
def combined_researcher(state: CombinedState):
    """Researcher with blackboard logging."""
    from datetime import datetime
    
    user_msg = state["messages"][0].content
    
    prompt = f"""{RESEARCHER_PROMPT}
Task: {user_msg}"""
    
    response = get_researcher_llm().invoke([HumanMessage(content=prompt)])
    findings = response.content
    
    # Log to blackboard
    blackboard.write(BlackboardEntry(
        agent="Researcher",
        timestamp=datetime.now().isoformat(),
        category="fact",
        content=findings,
        confidence=0.8
    ))
    
    blackboard.log_message(AgentMessage(
        msg_id="MSG-R1",
        sender="Researcher",
        recipient="Analyst",
        msg_type=MessageType.RESPONSE,
        topic="competitor research",
        payload=findings,
        confidence=0.8
    ))
    
    return {
        "research_findings": findings,
        "messages": [AIMessage(content=f"[Researcher] {findings}")]
    }


def combined_analyst(state: CombinedState):
    """Analyst with blackboard integration."""
    
    findings = state.get("research_findings", "")
    
    # Check if this is a revision (from debate loop)
    is_revision = state.get("debate_round", 0) > 1
    
    if is_revision:
        prompt = f"""{ANALYST_PROMPT}

REVISION REQUEST:
{state.get('critique_output', '')}

Here are the original findings:
{findings}

Revise your battlecard to address ALL issues raised. Return the COMPLETE revised battlecard."""
    else:
        prompt = f"""{ANALYST_PROMPT}

Here are the Researcher's findings:
{findings}

Synthesize these into a structured battlecard."""
    
    response = get_analyst_llm().invoke([HumanMessage(content=prompt)])
    analysis = response.content
    
    return {
        "analysis_output": analysis,
        "messages": [AIMessage(content=f"[Analyst] {analysis}")]
    }


def combined_critic(state: CombinedState):
    """Critic with blackboard integration."""
    
    analysis = state.get("analysis_output", "")
    findings = state.get("research_findings", "")
    
    round_num = state.get("debate_round", 1)
    max_rounds = state.get("max_debate_rounds", 3)
    
    prompt = f"""{CRITIC_PROMPT}

This is debate round {round_num} of {max_rounds}.

Analyst's battlecard:
{analysis}

Original findings:
{findings}

Provide your critique. At the end, explicitly state:
SATISFIED: YES or NO"""
    
    response = get_critic_llm().invoke([HumanMessage(content=prompt)])
    content = response.content
    
    satisfied = "SATISFIED: YES" in content.upper() or round_num >= max_rounds
    
    return {
        "critique_output": content,
        "critic_satisfied": satisfied,
        "debate_round": round_num + 1,
        "messages": [AIMessage(content=f"[Critic Round {round_num}] {content}")]
    }


def combined_finalizer(state: CombinedState):
    """Finalizer that synthesizes everything."""
    
    synthesis_prompt = f"""You are the Finalizer. Combine the Analyst's battlecard with the Critic's feedback.

BATTLECARD:
{state.get('analysis_output', '')}

CRITIQUE:
{state.get('critique_output', '')}

Instructions:
- Incorporate valid critiques into the final report
- Add a "Revision Notes" section explaining what changed
- Maintain source attribution for all claims
- Format as a professional markdown report"""
    
    response = get_finalizer_llm().invoke([HumanMessage(content=synthesis_prompt)])
    
    return {
        "final_report": response.content,
        "messages": [AIMessage(content=f"[Final Report]\n{response.content}")],
        "current_step": END
    }


# ============================================================
# ROUTING FUNCTIONS
# ============================================================
def combined_supervisor(state: CombinedState):
    """Supervisor that routes to the appropriate node."""
    step = state.get("current_step", "start")
    
    if step == "start":
        return {"current_step": "research"}
    elif step == "research":
        return {"current_step": "analysis"}
    elif step == "analysis":
        return {"current_step": "critique"}
    elif step == "critique":
        # Check if we need to loop back
        if state.get("critic_satisfied", False):
            return {"current_step": "finalize"}
        else:
            return {"current_step": "revision"}
    elif step == "revision":
        return {"current_step": "analysis"}  # Loop back to analyst for revision
    else:
        return {"current_step": END}


def route_from_supervisor(state: CombinedState):
    """Route from supervisor to the next node."""
    return state["current_step"]


# ============================================================
# BUILD THE COMPLETE SYSTEM
# ============================================================
def build_complete_system():
    """Build the complete multi-agent system."""
    
    builder = StateGraph(CombinedState)
    
    # Add nodes
    builder.add_node("supervisor", combined_supervisor)
    builder.add_node("researcher", combined_researcher)
    builder.add_node("analyst", combined_analyst)
    builder.add_node("critic", combined_critic)
    builder.add_node("finalizer", combined_finalizer)
    
    # Set entry point
    builder.set_entry_point("supervisor")
    
    # Conditional edges from supervisor
    builder.add_conditional_edges("supervisor", route_from_supervisor, {
        "research": "researcher",
        "analysis": "analyst",
        "critique": "critic",
        "revision": "analyst",
        "finalize": "finalizer",
        END: END
    })
    
    # After each worker, return to supervisor
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("critic", "supervisor")
    builder.add_edge("finalizer", END)
    
    return builder.compile()


# ============================================================
# RUN THE COMPLETE SYSTEM
# ============================================================
if __name__ == "__main__":
    graph = build_complete_system()
    
    # Clear blackboard
    blackboard.clear()
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Research Tesla and produce a competitive battlecard")],
        "current_step": "start",
        "debate_round": 1,
        "max_debate_rounds": 3,
        "critic_satisfied": False
    }
    
    # Run the system
    result = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result.get("final_report", "No report generated"))
    
    # Summary
    print("\n" + "=" * 60)
    print("SYSTEM SUMMARY")
    print("=" * 60)
    print(f"Debate Rounds: {result.get('debate_round', 0) - 1}")
    print(f"Critic Satisfied: {result.get('critic_satisfied', False)}")
    
    # Blackboard summary
    summary = blackboard.get_summary()
    print(f"\nBlackboard:")
    print(f"  Total Entries: {summary['total_entries']}")
    print(f"  Facts: {summary['facts']}")
    print(f"  Insights: {summary['insights']}")
    print(f"  Critiques: {summary['critiques']}")
    print(f"  Conflicts: {summary['conflicts']}")
    print(f"  Agents: {', '.join(summary['agents'])}")