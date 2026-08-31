"""
07_transcript.py
Handles exporting transcripts and audit logs.
"""

import json
from datetime import datetime

from 05_blackboard import blackboard
from 04_message_protocol import AgentMessage


def export_transcript_markdown(filename: str = "agent_transcript.md") -> str:
    """Export the full transcript as markdown."""
    transcript = blackboard.export_transcript()
    
    with open(filename, "w") as f:
        f.write(transcript)
    
    print(f"✅ Transcript exported to {filename}")
    return transcript


def export_transcript_json(filename: str = "agent_transcript.json") -> str:
    """Export the full transcript as JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "messages": [
            {
                "msg_id": m.msg_id,
                "sender": m.sender,
                "recipient": m.recipient,
                "msg_type": m.msg_type.value,
                "topic": m.topic,
                "payload": m.payload,
                "references": m.references,
                "confidence": m.confidence,
                "timestamp": m.timestamp
            }
            for m in blackboard.agent_messages
        ],
        "blackboard": [
            {
                "agent": e.agent,
                "category": e.category,
                "content": e.content,
                "sources": e.sources,
                "confidence": e.confidence,
                "timestamp": e.timestamp
            }
            for e in blackboard.entries
        ]
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ JSON transcript exported to {filename}")
    return filename


def print_transcript_summary():
    """Print a summary of the transcript."""
    summary = blackboard.get_summary()
    
    print("\n📊 TRANSCRIPT SUMMARY")
    print("=" * 40)
    print(f"Total Messages: {summary['total_messages']}")
    print(f"Total Entries: {summary['total_entries']}")
    print(f"Facts: {summary['facts']}")
    print(f"Insights: {summary['insights']}")
    print(f"Critiques: {summary['critiques']}")
    print(f"Conflicts: {summary['conflicts']}")
    print(f"Agents: {', '.join(summary['agents'])}")


def generate_audit_report() -> str:
    """Generate a complete audit report."""
    lines = []
    lines.append("# AGENT AUDIT REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    
    # Agent communication summary
    lines.append("## Agent Communication")
    for msg in blackboard.agent_messages:
        lines.append(f"- [{msg.msg_id}] {msg.sender} → {msg.recipient} ({msg.msg_type.value}): {msg.topic}")
    
    lines.append("")
    lines.append("## Blackboard Contents")
    for entry in blackboard.entries:
        lines.append(f"- **{entry.agent}** [{entry.category}]")
        lines.append(f"  {entry.content[:150]}...")
        if entry.sources:
            lines.append(f"  Sources: {', '.join(entry.sources)}")
        lines.append(f"  Confidence: {entry.confidence}")
    
    # Conflicts
    conflicts = blackboard.get_conflicting_entries()
    if conflicts:
        lines.append("")
        lines.append("## ⚠️ Conflicts Detected")
        for a, b in conflicts:
            lines.append(f"- {a.agent}: {a.content[:50]}... vs {b.agent}: {b.content[:50]}...")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Add some test data
    from 05_blackboard import BlackboardEntry
    
    blackboard.clear()
    
    blackboard.log_message(AgentMessage(
        msg_id="MSG-001",
        sender="Researcher",
        recipient="Analyst",
        msg_type="response",
        topic="Tesla Revenue",
        payload="Tesla Q3 2024 revenue: $25.18B"
    ))
    
    blackboard.write(BlackboardEntry(
        agent="Researcher",
        category="fact",
        content="Tesla Q3 2024 revenue: $25.18B",
        confidence=0.9
    ))
    
    # Export transcripts
    export_transcript_markdown()
    export_transcript_json()
    print_transcript_summary()
    
    # Generate audit report
    report = generate_audit_report()
    print("\n" + report)