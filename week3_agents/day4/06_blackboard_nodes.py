"""
06_blackboard_nodes.py
Agent nodes that read from and write to the blackboard.
"""

from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

from 01_agent_roles import (
    RESEARCHER_PROMPT, ANALYST_PROMPT, CRITIC_PROMPT,
    get_researcher_llm, get_analyst_llm, get_critic_llm
)
from 04_message_protocol import AgentMessage, MessageType
from 05_blackboard import BlackboardEntry, blackboard


# ============================================================
# RESEARCHER WITH BLACKBOARD
# ============================================================
def researcher_with_blackboard(state: dict) -> dict:
    """Researcher that writes findings to the blackboard."""
    
    user_msg = state["messages"][0].content
    
    prompt = f"""{RESEARCHER_PROMPT}
Task: {user_msg}"""
    
    response = get_researcher_llm().invoke([HumanMessage(content=prompt)])
    findings = response.content
    
    # Write findings to blackboard
    blackboard.write(BlackboardEntry(
        agent="Researcher",
        timestamp=datetime.now().isoformat(),
        category="fact",
        content=findings,
        sources=["web_search"],
        confidence=0.8
    ))
    
    # Also log as a message
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


# ============================================================
# ANALYST WITH BLACKBOARD
# ============================================================
def analyst_with_blackboard(state: dict) -> dict:
    """Analyst that reads from blackboard and writes insights."""
    
    # Read all facts from the blackboard
    facts = blackboard.read_by_category("fact")
    fact_text = "\n\n".join([f"[{f.agent}] {f.content}" for f in facts])
    
    prompt = f"""{ANALYST_PROMPT}

Facts from blackboard:
{fact_text}

Synthesize into battlecard."""
    
    response = get_analyst_llm().invoke([HumanMessage(content=prompt)])
    analysis = response.content
    
    blackboard.write(BlackboardEntry(
        agent="Analyst",
        category="insight",
        content=analysis,
        confidence=0.75
    ))
    
    return {
        "analysis_output": analysis,
        "messages": [AIMessage(content=f"[Analyst] {analysis}")]
    }


# ============================================================
# CRITIC WITH BLACKBOARD
# ============================================================
def critic_with_blackboard(state: dict) -> dict:
    """Critic that reads from blackboard and writes critiques."""
    
    # Read insights from the blackboard
    insights = blackboard.read_by_category("insight")
    insight_text = "\n\n".join([f"[{i.agent}] {i.content}" for i in insights])
    
    facts = blackboard.read_by_category("fact")
    fact_text = "\n\n".join([f"[{f.agent}] {f.content}" for f in facts])
    
    prompt = f"""{CRITIC_PROMPT}

Insights to review:
{insight_text}

Original facts:
{fact_text}

Provide your critique. Identify at least 3 specific issues."""
    
    response = get_critic_llm().invoke([HumanMessage(content=prompt)])
    critique = response.content
    
    blackboard.write(BlackboardEntry(
        agent="Critic",
        category="critique",
        content=critique,
        confidence=0.85
    ))
    
    return {
        "critique_output": critique,
        "messages": [AIMessage(content=f"[Critic] {critique}")]
    }


# ============================================================
# CHECK CONFLICTS
# ============================================================
def check_conflicts() -> list:
    """Check for conflicts on the blackboard."""
    conflicts = blackboard.get_conflicting_entries()
    
    if conflicts:
        print(f"\n⚠️ Found {len(conflicts)} conflict(s):")
        for a, b in conflicts:
            print(f"  - {a.agent}: {a.content[:50]}...")
            print(f"  - {b.agent}: {b.content[:50]}...")
    else:
        print("\n✅ No conflicts found")
    
    return conflicts


# ============================================================
# DEMO
# ============================================================
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    # Clear the blackboard
    blackboard.clear()
    
    print("=" * 60)
    print("RUNNING AGENTS WITH BLACKBOARD")
    print("=" * 60)
    
    # Simulate the conversation
    state = {"messages": [HumanMessage(content="Research Tesla")]}
    
    # Run each agent
    state = researcher_with_blackboard(state)
    state = analyst_with_blackboard(state)
    state = critic_with_blackboard(state)
    
    # Check for conflicts
    check_conflicts()
    
    # Export transcript
    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(blackboard.export_transcript())
    
    # Print summary
    summary = blackboard.get_summary()
    print(f"\n📊 Summary: {summary}")