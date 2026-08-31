"""
guardrails.py
Prevents Apex from taking harmful, costly, or incorrect actions
(Week 3, Skill 1, Concept 7: Guardrails & Safe Tool Execution).

"Always assume the agent will try something unexpected" - every guardrail
here is a hard, code-level check, not something the LLM is merely asked
nicely to respect.
"""

# --- Allow-list: only these tools may ever be called ---
ALLOWED_TOOLS = {
    "search_web_leads", "search_real_leads", "enrich_company", "check_crm_duplicate",
    "query_similar_deals", "lookup_case_study", "check_pricing_faq",
    "book_meeting", "send_outreach_message",
}

# --- Cost guardrail ---
MAX_COST_PER_LEAD_USD = 0.50

# --- Negotiation guardrail ---
MAX_AUTONOMOUS_DISCOUNT_PCT = 15  # anything above this requires human approval

# --- Loop guardrails ---
MAX_CLOSE_ITERATIONS = 6
MAX_SOURCE_RETRIES = 2
MAX_WEB_SEARCHES_PER_SOURCE_RUN = 3  # caps DDG calls per single source-node run

# --- Simple PII pattern flags (deliberately conservative - false positives are safer than misses) ---
import re
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),          # SSN-like
    re.compile(r"\b\d{16}\b"),                      # raw card-number-like
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone-like (only flagged for OUTBOUND external tools)
]


def is_tool_allowed(tool_name: str) -> bool:
    return tool_name in ALLOWED_TOOLS


def check_cost_guardrail(state) -> bool:
    """Returns True if the lead is still under its cost cap."""
    return state.get("total_estimated_cost_usd", 0.0) < MAX_COST_PER_LEAD_USD


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def discount_requires_approval(offer_pct: float) -> bool:
    return offer_pct > MAX_AUTONOMOUS_DISCOUNT_PCT


def record_cost(state, estimated_usd: float) -> float:
    """Returns the new running total - callers merge this into state."""
    return state.get("total_estimated_cost_usd", 0.0) + estimated_usd