"""
hitl.py
Human-in-the-loop checkpoint handling (Week 3, Skill 1, Concept 8).

When close_node or source_node sets needs_human_approval=True, the graph
run ends there (see graph.py's routing) and control returns to the caller
(app.py). The pending_action dict always has three parts, matching the
study guide's format exactly:
    - action:      what Apex wants to do
    - reasoning:   why it wants to do it
    - consequence: what happens if approved vs. rejected

hitl_node() is a graph node like any other: it reads state["human_decision"]
(set by app.py BEFORE re-invoking the graph with the same thread_id) and
resolves it into the next state - approving, rejecting, or applying a
human's own written reply.
"""

from guardrails import MAX_AUTONOMOUS_DISCOUNT_PCT
from tools import execute_tool


def hitl_node(state: dict) -> dict:
    decision = state.get("human_decision")
    feedback = state.get("human_feedback")
    lead = state.get("lead", {})
    pending = state.get("pending_action") or {}

    if decision is None:
        # Nothing to resolve yet - the graph shouldn't route here without a
        # decision, but fail safely rather than crash if it ever does.
        return {}

    base_clear = {"needs_human_approval": False, "pending_action": None,
                  "human_decision": None, "human_feedback": None}

    action_text = pending.get("action", "")

    # --- Discount escalation ---
    if "discount" in action_text.lower():
        if decision == "approve":
            new_history = [{"role": "apex", "content": feedback or "Confirming the approved discount - looking forward to moving ahead together."}]
            return {**base_clear, "deal_stage": "Negotiating", "conversation_history": new_history}
        else:
            hold_firm = (
                f"After review, the best we can offer is {MAX_AUTONOMOUS_DISCOUNT_PCT}% - "
                "hoping that still works for you."
            )
            return {**base_clear, "deal_stage": "Negotiating",
                    "negotiation_offer_pct": MAX_AUTONOMOUS_DISCOUNT_PCT,
                    "conversation_history": [{"role": "apex", "content": hold_firm}]}

    # --- Hostile / stalled engagement takeover ---
    if feedback:
        # A human wrote their own reply - inject it directly and let Apex resume next turn.
        return {**base_clear, "deal_stage": "Negotiating",
                "conversation_history": [{"role": "human", "content": feedback}]}

    # Escalation acknowledged with no reply text yet (human is taking over silently)
    return {**base_clear, "deal_stage": "Escalated" if decision == "reject" else "Negotiating"}