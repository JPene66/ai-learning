"""
state.py
The shared state schema that flows through every node in Apex's graph
(Week 3, Skill 1, Concept 9: State Management & Agent Loop Control).

Fields using Annotated[..., operator.add] are REDUCERS - LangGraph appends
to these instead of overwriting them, so every node can add a log entry
without needing to know what previous nodes already logged. This is the
exact "reducers" concept from the study guide's Wednesday session.

This single dict is what gets checkpointed to disk after every node runs -
it IS the "persistent state" the project brief requires. Killing the
process mid-run and re-invoking with the same thread_id resumes from here.
"""

import operator
from typing import TypedDict, Annotated, Optional, List, Dict, Any


class AgentState(TypedDict):
    # --- Identity & pipeline position ---
    lead_id: str
    deal_stage: str  # "Sourced" | "Qualified" | "Disqualified" | "Engaging" | "Negotiating" | "Booked" | "Escalated" | "Lost"

    # --- Source stage output ---
    lead: Dict[str, Any]                 # the current lead's firmographic data
    country: Optional[str]               # target country for sourcing (e.g. "Nigeria")
    region: Optional[str]                # target region/city (e.g. "Lagos")
    source_max_leads: int                # how many real leads to fetch (1-5, default 2)
    source_channels_tried: Annotated[List[str], operator.add]
    source_confidence: float             # used by the CRAG-style fallback check

    # --- Qualify stage output ---
    bant_medicc: Dict[str, Any]          # per-criterion scores + rationale
    qualification_score: float           # 0-100
    segment: str                         # "Hot" | "Warm" | "Cold" | "Disqualified"

    # --- Close stage / memory ---
    conversation_history: Annotated[List[Dict[str, str]], operator.add]  # short-term memory (this lead only)
    objection_log: Annotated[List[Dict[str, Any]], operator.add]         # episodic memory (lessons learned)
    outreach_draft: Optional[str]
    latest_lead_message: Optional[str]   # what the lead just said - drives this turn's Close action
    negotiation_offer_pct: float         # current discount % on the table, for guardrail checks

    # --- Loop / cost control ---
    close_iterations: int
    total_estimated_cost_usd: float

    # --- HITL ---
    needs_human_approval: bool
    pending_action: Optional[Dict[str, Any]]  # {"action": ..., "reasoning": ..., "consequence": ...}
    human_decision: Optional[str]             # "approve" | "reject" | "edit"
    human_feedback: Optional[str]

    # --- Reliability ---
    error_log: Annotated[List[str], operator.add]
    retry_count: int


def new_lead_state(lead_id: str, initial_lead: Optional[Dict[str, Any]] = None) -> AgentState:
    """Factory for a fresh state when starting a new lead through the pipeline."""
    return AgentState(
        lead_id=lead_id,
        deal_stage="New",
        lead=initial_lead or {},
        country=initial_lead.get("country") if initial_lead else None,
        region=initial_lead.get("region") if initial_lead else None,
        source_max_leads=initial_lead.get("max_leads", 2) if initial_lead else 2,
        source_channels_tried=[],
        source_confidence=0.0,
        bant_medicc={},
        qualification_score=0.0,
        segment="Unscored",
        conversation_history=[],
        objection_log=[],
        outreach_draft=None,
        latest_lead_message=None,
        negotiation_offer_pct=0.0,
        close_iterations=0,
        total_estimated_cost_usd=0.0,
        needs_human_approval=False,
        pending_action=None,
        human_decision=None,
        human_feedback=None,
        error_log=[],
        retry_count=0,
    )