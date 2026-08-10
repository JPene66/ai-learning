"""
04_message_protocol.py
Defines the structured message protocol for inter-agent communication.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional, List
from enum import Enum


class MessageType(str, Enum):
    """Types of messages between agents."""
    REQUEST = "request"          # "Please do X"
    RESPONSE = "response"        # "Here is the result of X"
    CRITIQUE = "critique"        # "I challenge your claim about X"
    NOTIFICATION = "notification"  # "FYI: X happened"
    DELEGATION = "delegation"    # "Supervisor assigns task X to Agent Y"


class AgentMessage(BaseModel):
    """Structured message between agents."""
    
    msg_id: str = Field(description="Unique message ID, e.g., MSG-001")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of the message"
    )
    sender: str = Field(
        description="Agent name: 'Researcher', 'Analyst', 'Critic', 'Supervisor'"
    )
    recipient: str = Field(
        description="Target agent or 'ALL' for broadcast"
    )
    msg_type: MessageType
    topic: str = Field(description="What this message is about")
    payload: str = Field(description="The actual content")
    references: List[str] = Field(
        default_factory=list,
        description="IDs of messages this refers to"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="0.0-1.0, if applicable"
    )
    
    def to_markdown(self) -> str:
        """Format for human-readable logs."""
        return f"""### [{self.msg_type.upper()}] {self.topic}
**From:** {self.sender} → **To:** {self.recipient}
**Time:** {self.timestamp}
**Confidence:** {self.confidence or 'N/A'}

{self.payload}

---"""


# ============================================================
# MESSAGE FACTORIES
# ============================================================
def create_request(sender: str, recipient: str, topic: str, payload: str) -> AgentMessage:
    """Create a request message."""
    return AgentMessage(
        msg_id=f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(sender)}",
        sender=sender,
        recipient=recipient,
        msg_type=MessageType.REQUEST,
        topic=topic,
        payload=payload
    )


def create_response(sender: str, recipient: str, topic: str, payload: str, 
                    references: List[str] = None) -> AgentMessage:
    """Create a response message."""
    return AgentMessage(
        msg_id=f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(sender)}",
        sender=sender,
        recipient=recipient,
        msg_type=MessageType.RESPONSE,
        topic=topic,
        payload=payload,
        references=references or []
    )


def create_critique(sender: str, recipient: str, topic: str, payload: str,
                    references: List[str] = None) -> AgentMessage:
    """Create a critique message."""
    return AgentMessage(
        msg_id=f"MSG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(sender)}",
        sender=sender,
        recipient=recipient,
        msg_type=MessageType.CRITIQUE,
        topic=topic,
        payload=payload,
        references=references or []
    )


# ============================================================
# TEST THE MESSAGE PROTOCOL
# ============================================================
if __name__ == "__main__":
    msg = create_request(
        sender="Supervisor",
        recipient="Researcher",
        topic="Tesla Research",
        payload="Please research Tesla's financial performance for Q3 2024."
    )
    
    print(msg.to_markdown())