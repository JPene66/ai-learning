"""
close_agent.py
The Close node: "Engages qualified leads via personalized outreach with
negotiation capabilities and booking handoff."

Each invocation of this node handles exactly ONE turn of the conversation -
either drafting the opening outreach (if none exists yet) or reacting to
`state["latest_lead_message"]`. This is deliberate: it maps the ReAct loop
(Concept 1: reason about the message -> act on it -> the graph pauses ->
a new turn arrives) directly onto LangGraph's checkpoint/resume model,
which is also what makes "persistent state" and "human-in-the-loop" work
naturally rather than needing bespoke pause/resume plumbing.

Demonstrates:
  - ReAct pattern (classify intent -> act accordingly)
  - Long-term + episodic memory (retrieval grounds responses; lessons get recorded)
  - Iterative retrieve-refine loop for objections (retrieval.py)
  - Negotiation guardrails (discount cap -> auto-handle or escalate)
  - Human-in-the-loop escalation (hostile sentiment, discounts over the cap,
    exceeding the autonomous turn limit)
  - Booking handoff (tool call)
"""

from helpers import ask, ask_json
from retrieval import retrieve_refine_objection_response, get_pricing_answer
from memory import get_long_term_memory, seed_long_term_memory, trim_short_term_memory, record_episodic_lesson
from knowledge_base import CASE_STUDIES
from tools import execute_tool
from guardrails import (
    MAX_CLOSE_ITERATIONS, discount_requires_approval, record_cost,
)

OBJECTION_TOPICS = ["pricing", "contract_length", "security_compliance", "competitor_comparison"]


def _find_case_study(industry: str):
    matches = [c for c in CASE_STUDIES if c["industry"] == industry]
    return matches[0] if matches else CASE_STUDIES[0]


def _draft_opening_outreach(lead: dict) -> str:
    case_study = _find_case_study(lead.get("industry", ""))
    prompt = f"""Write a short (3-4 sentence), warm but professional cold outreach message to
{lead.get('company', 'this prospect')}, a company in the {lead.get('industry', 'unknown')} industry
with a known pain point of "{lead.get('known_pain_point', 'operational inefficiency')}".

Reference this relevant case study naturally, without sounding like a canned pitch:
{case_study['text']}

End with a soft call to action to book a short call."""
    return ask(prompt, max_tokens=180)


def _classify_turn(message: str) -> dict:
    """One classification call covering sentiment, intent, and any discount
    ask - kept on the cheap ROUTING_MODEL since this runs every single turn."""
    prompt = f"""Classify this message from a sales prospect.

Message: "{message}"

Respond ONLY as JSON:
{{"sentiment": "positive|neutral|hostile",
  "intent": "objection|discount_request|agreement|question|other",
  "objection_topic": one of {OBJECTION_TOPICS} or null,
  "requested_discount_pct": number or null}}"""
    return ask_json(prompt)


def _handle_objection(state: dict, message: str, objection_topic: str) -> dict:
    collection = get_long_term_memory()
    seed_long_term_memory(collection)

    quick_answer = get_pricing_answer(objection_topic)
    response, iterations, sources = retrieve_refine_objection_response(message, collection, max_iterations=2)
    final_response = response if not quick_answer else f"{quick_answer} {response}"

    record_episodic_lesson(state["lead_id"], objection_topic or "general", final_response)

    return {
        "response": final_response,
        "objection_entry": {"topic": objection_topic or "general", "message": message, "resolution": final_response, "sources": sources},
        "cost_delta": 0.02 * iterations,
    }


def _handle_discount_request(requested_pct: float, lead: dict) -> dict:
    if discount_requires_approval(requested_pct):
        return {
            "needs_escalation": True,
            "pending_action": {
                "action": f"Approve a {requested_pct}% discount for {lead.get('company', 'this lead')}",
                "reasoning": f"Requested discount ({requested_pct}%) exceeds Apex's autonomous limit.",
                "consequence": "If approved, Apex will offer this discount and continue negotiating; if rejected, Apex will hold firm at the maximum autonomous discount.",
            },
        }
    response = ask(
        f"Write a short, confident 2-sentence reply confirming a {requested_pct}% discount "
        f"is approved for {lead.get('company', 'this account')}, framed as a gesture of good faith for a strong partnership.",
        max_tokens=100,
    )
    return {"needs_escalation": False, "response": response, "offer_pct": requested_pct}


def close_node(state: dict) -> dict:
    lead = state.get("lead", {})
    message = state.get("latest_lead_message")
    history = state.get("conversation_history", [])
    iterations = state.get("close_iterations", 0) + 1

    if iterations > MAX_CLOSE_ITERATIONS:
        return {
            "needs_human_approval": True,
            "pending_action": {
                "action": f"Review stalled engagement with {lead.get('company', 'this lead')}",
                "reasoning": f"Reached the {MAX_CLOSE_ITERATIONS}-turn autonomous engagement limit with no resolution.",
                "consequence": "Apex will pause outreach until a human reviews the full conversation.",
            },
            "deal_stage": "Escalated",
            "close_iterations": iterations,
        }

    if not message and not history:
        draft = _draft_opening_outreach(lead)
        return {
            "outreach_draft": draft,
            "conversation_history": [{"role": "apex", "content": draft}],
            "deal_stage": "Engaging",
            "close_iterations": iterations,
            "total_estimated_cost_usd": record_cost(state, 0.01),
        }

    if not message:
        return {"close_iterations": iterations}

    classification = _classify_turn(message)
    new_history = [{"role": "lead", "content": message}]

    if classification.get("sentiment") == "hostile":
        return {
            "conversation_history": new_history,
            "needs_human_approval": True,
            "pending_action": {
                "action": f"Take over conversation with {lead.get('company', 'this lead')}",
                "reasoning": "Message was classified as hostile in tone.",
                "consequence": "Apex will pause all outreach to this lead until a human responds directly.",
            },
            "deal_stage": "Escalated",
            "close_iterations": iterations,
        }

    intent = classification.get("intent", "other")

    if intent == "discount_request" and classification.get("requested_discount_pct"):
        outcome = _handle_discount_request(classification["requested_discount_pct"], lead)
        if outcome["needs_escalation"]:
            return {
                "conversation_history": new_history,
                "needs_human_approval": True,
                "pending_action": outcome["pending_action"],
                "deal_stage": "Escalated",
                "close_iterations": iterations,
            }
        new_history.append({"role": "apex", "content": outcome["response"]})
        return {
            "conversation_history": new_history,
            "negotiation_offer_pct": outcome["offer_pct"],
            "deal_stage": "Negotiating",
            "close_iterations": iterations,
            "total_estimated_cost_usd": record_cost(state, 0.01),
        }

    if intent == "objection":
        outcome = _handle_objection(state, message, classification.get("objection_topic"))
        new_history.append({"role": "apex", "content": outcome["response"]})
        return {
            "conversation_history": new_history,
            "objection_log": [outcome["objection_entry"]],
            "deal_stage": "Negotiating",
            "close_iterations": iterations,
            "total_estimated_cost_usd": record_cost(state, outcome["cost_delta"]),
        }

    if intent == "agreement":
        ok, booking = execute_tool("book_meeting", {"company": lead.get("company", "Prospect"), "requested_day": "next available slot"})
        confirmation = (
            f"Great news - you're booked in! Confirmation: {booking['confirmation_id']}."
            if ok else "I'll have someone reach out to finalize the meeting time."
        )
        new_history.append({"role": "apex", "content": confirmation})
        return {
            "conversation_history": new_history,
            "deal_stage": "Booked",
            "close_iterations": iterations,
            "total_estimated_cost_usd": record_cost(state, 0.01),
        }

    trimmed = trim_short_term_memory(history)
    context = "\n".join(f"{m['role']}: {m['content']}" for m in trimmed)
    reply = ask(f"Conversation so far:\n{context}\n\nLead just said: {message}\n\nReply helpfully and briefly, staying in a sales-conversation tone.", max_tokens=150)
    new_history.append({"role": "apex", "content": reply})
    return {
        "conversation_history": new_history,
        "close_iterations": iterations,
        "total_estimated_cost_usd": record_cost(state, 0.01),
    }