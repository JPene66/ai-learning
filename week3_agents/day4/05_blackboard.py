"""
05_blackboard.py
Implements the Blackboard (shared workspace) for all agents.
"""

from typing import List, Dict, Literal, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

from 04_message_protocol import AgentMessage


class BlackboardEntry(BaseModel):
    """A single entry on the blackboard."""
    
    agent: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    category: Literal["fact", "insight", "critique", "decision", "uncertainty"]
    content: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.5


class Blackboard:
    """Shared workspace for all agents."""
    
    def __init__(self):
        self.entries: List[BlackboardEntry] = []
        self.agent_messages: List[AgentMessage] = []
    
    def write(self, entry: BlackboardEntry):
        """Any agent can write a finding to the blackboard."""
        self.entries.append(entry)
        print(f"📝 Blackboard: {entry.agent} added {entry.category} — {entry.content[:60]}...")
    
    def read_by_category(self, category: str) -> List[BlackboardEntry]:
        """Agents read relevant entries by category."""
        return [e for e in self.entries if e.category == category]
    
    def read_by_agent(self, agent: str) -> List[BlackboardEntry]:
        """Read entries by a specific agent."""
        return [e for e in self.entries if e.agent == agent]
    
    def read_all(self) -> List[BlackboardEntry]:
        """Read all entries."""
        return self.entries
    
    def get_conflicting_entries(self) -> List[Tuple[BlackboardEntry, BlackboardEntry]]:
        """Find entries where agents disagree."""
        conflicts = []
        facts = [e for e in self.entries if e.category == "fact"]
        
        for i, a in enumerate(facts):
            for b in facts[i+1:]:
                if a.content != b.content and len(a.content) > 10:
                    conflicts.append((a, b))
        
        return conflicts
    
    def log_message(self, msg: AgentMessage):
        """Log an inter-agent message."""
        self.agent_messages.append(msg)
        print(f"📨 Logged message: {msg.sender} → {msg.recipient} [{msg.msg_type.value}]")
    
    def export_transcript(self) -> str:
        """Export all messages as a markdown transcript for audit."""
        lines = ["# Agent Communication Transcript\n"]
        
        for msg in self.agent_messages:
            lines.append(msg.to_markdown())
        
        # Add blackboard entries
        if self.entries:
            lines.append("\n## Blackboard Summary\n")
            for entry in self.entries:
                lines.append(f"- **{entry.agent}** ({entry.category}): {entry.content[:100]}...")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear the blackboard."""
        self.entries = []
        self.agent_messages = []
    
    def get_summary(self) -> dict:
        """Get a summary of the blackboard contents."""
        return {
            "total_entries": len(self.entries),
            "total_messages": len(self.agent_messages),
            "facts": len(self.read_by_category("fact")),
            "insights": len(self.read_by_category("insight")),
            "critiques": len(self.read_by_category("critique")),
            "conflicts": len(self.get_conflicting_entries()),
            "agents": list(set(e.agent for e in self.entries))
        }


# Create a global blackboard instance
blackboard = Blackboard()


# ============================================================
# TEST THE BLACKBOARD
# ============================================================
if __name__ == "__main__":
    # Test writing and reading
    blackboard.write(BlackboardEntry(
        agent="Researcher",
        category="fact",
        content="Tesla Q3 2024 revenue: $25.18B",
        sources=["Tesla 10-Q"],
        confidence=0.9
    ))
    
    blackboard.write(BlackboardEntry(
        agent="Analyst",
        category="insight",
        content="Tesla's revenue growth is strong",
        sources=["Researcher"],
        confidence=0.8
    ))
    
    facts = blackboard.read_by_category("fact")
    print(f"Facts on blackboard: {len(facts)}")
    
    summary = blackboard.get_summary()
    print(f"\nBlackboard Summary: {summary}")