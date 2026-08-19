"""
source_agent.py
The Source node: "Finds and enriches leads from multiple channels."

Demonstrates:
  - Query planning & decomposition (retrieval.plan_source_queries)
  - Multi-source retrieval (web search + CRM duplicate check, routed per sub-query)
  - Tool-augmented retrieval (every data-gathering step is a tool call, not a raw LLM guess)
  - Corrective RAG / fallback: if the first pass finds nothing usable, broaden
    the search and retry, up to a guardrail-capped number of attempts
  - Cost guardrail tracking
"""

from retrieval import plan_source_queries, route_query
from tools import execute_tool
from guardrails import MAX_SOURCE_RETRIES, check_cost_guardrail, record_cost

ESTIMATED_COST_PER_TOOL_CALL = 0.01  # rough placeholder cost per tool/LLM round for guardrail tracking


def _gather_candidates(industry: str, max_results: int = 3):
    ok, result = execute_tool("search_web_leads", {"industry": industry, "max_results": max_results})
    return result if ok else []


def _dedupe_and_enrich(candidates: list):
    enriched = []
    for c in candidates:
        ok_dup, dup_result = execute_tool("check_crm_duplicate", {"company": c["company"]})
        if ok_dup and dup_result.get("is_duplicate"):
            continue  # skip - already a known account, not a fresh lead
        ok_enrich, enrich_result = execute_tool("enrich_company", {"company": c["company"]})
        if ok_enrich:
            enriched.append({**c, **enrich_result})
    return enriched


def source_node(state: dict) -> dict:
    """
    LangGraph node function: reads state, does work, returns a PARTIAL
    state update dict (LangGraph merges this into the full state - it does
    not need to return every field, just what changed).
    """
    lead = state.get("lead", {})
    industry = lead.get("industry", "fintech")  # default target industry if none specified yet
    goal = f"Find promising {industry} leads similar to our best past customers"

    sub_queries = plan_source_queries(goal)
    channels_tried = [route_query(q) for q in sub_queries] or ["web"]

    candidates = _gather_candidates(industry, max_results=3)
    enriched = _dedupe_and_enrich(candidates)

    attempts = 1
    # Corrective fallback: broaden search if the first pass yields nothing usable
    while not enriched and attempts <= MAX_SOURCE_RETRIES:
        broader_industry = "fintech" if industry != "fintech" else "logistics"
        candidates = _gather_candidates(broader_industry, max_results=3)
        enriched = _dedupe_and_enrich(candidates)
        attempts += 1

    confidence = min(1.0, len(enriched) / 3) if enriched else 0.0
    chosen_lead = max(enriched, key=lambda c: c.get("estimated_budget_usd", 0)) if enriched else lead

    new_cost = record_cost(state, ESTIMATED_COST_PER_TOOL_CALL * (len(sub_queries) + attempts * 2))

    error_log = []
    if not enriched:
        error_log.append("Source stage exhausted retries with no qualifying leads found.")

    return {
        "lead": {**lead, **chosen_lead},
        "source_channels_tried": channels_tried,
        "source_confidence": confidence,
        "deal_stage": "Sourced" if enriched else "Lost",
        "total_estimated_cost_usd": new_cost,
        "error_log": error_log,
    }