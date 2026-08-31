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

Real search flow (replaces mock data):
  1. Call search_real_leads() with industry, country, region and max_results.
  2. De-duplicate against the CRM and enrich each candidate.
  3. If the first pass is empty, retry with progressively broader geography
     (drop the region, then try the whole country), up to MAX_SOURCE_RETRIES.
"""

from retrieval import plan_source_queries, route_query
from tools import execute_tool
from guardrails import MAX_SOURCE_RETRIES, check_cost_guardrail, record_cost

# GPT parse call in search_real_leads is the meaningful cost per run
ESTIMATED_COST_PER_TOOL_CALL = 0.005


def _gather_real_candidates(industry: str, country: str, region: str = "",
                             max_results: int = 2) -> list:
    """Call the real DuckDuckGo-powered search tool."""
    ok, result = execute_tool("search_real_leads", {
        "industry": industry,
        "country": country,
        "region": region,
        "max_results": max_results,
    })
    return result if ok and isinstance(result, list) else []


def _dedupe_and_enrich(candidates: list) -> list:
    enriched = []
    for c in candidates:
        ok_dup, dup_result = execute_tool("check_crm_duplicate", {"company": c["company"]})
        if ok_dup and dup_result.get("is_duplicate"):
            continue  # skip — already a known account, not a fresh lead
        ok_enrich, enrich_result = execute_tool("enrich_company", {"company": c["company"]})
        if ok_enrich:
            # Merge enrichment data but keep the real-search fields (country, region, description)
            enriched.append({**c, **enrich_result})
        else:
            # Still add the candidate without enrichment if enrichment fails
            enriched.append(c)
    return enriched


def source_node(state: dict) -> dict:
    """
    LangGraph node function: reads state, does work, returns a PARTIAL
    state update dict (LangGraph merges this into the full state — it does
    not need to return every field, just what changed).
    """
    lead = state.get("lead", {})

    # Resolve fields from state first, then fall back to lead dict, then defaults.
    # This defensive chain ensures schema migrations (new fields added to state)
    # never silently break existing checkpointed threads.
    industry    = (state.get("country") and lead.get("industry")) or lead.get("industry") or "Fintech"
    country     = state.get("country") or lead.get("country") or "Nigeria"
    region      = state.get("region")  or lead.get("region")  or ""
    max_results = state.get("source_max_leads") or lead.get("max_leads") or 2

    print(f"[source_node] Searching: industry={industry!r}, country={country!r}, region={region!r}, max={max_results}")

    goal = f"Find promising {industry} leads in {region + ', ' if region else ''}{country}"
    sub_queries = plan_source_queries(goal)
    channels_tried = [route_query(q) for q in sub_queries] or ["web"]

    # --- First pass: real search with full geography ---
    candidates = _gather_real_candidates(industry, country, region, max_results)
    enriched   = _dedupe_and_enrich(candidates)

    attempts = 1

    # --- Corrective fallback: broaden geography progressively ---
    while not enriched and attempts <= MAX_SOURCE_RETRIES:
        if attempts == 1 and region:
            # Retry without region (country-wide)
            candidates = _gather_real_candidates(industry, country, "", max_results)
        else:
            # Last resort: drop region and country specifics, just use industry
            candidates = _gather_real_candidates(industry, country, "", max_results + 1)
        enriched = _dedupe_and_enrich(candidates)
        attempts += 1

    confidence  = min(1.0, len(enriched) / max(max_results, 1)) if enriched else 0.0
    # Pick the lead with the highest estimated budget; fall back to initial if none found
    chosen_lead = max(enriched, key=lambda c: c.get("estimated_budget_usd", 0)) if enriched else lead

    # Cost: DDG is free; GPT parse call per attempt is ~$0.005
    new_cost = record_cost(state, ESTIMATED_COST_PER_TOOL_CALL * attempts)

    error_log = []
    if not enriched:
        error_log.append(
            f"Source stage exhausted {attempts - 1} retries searching for "
            f"{industry} leads in {region + ', ' if region else ''}{country} — no results found."
        )

    return {
        "lead": {**lead, **chosen_lead},
        "source_channels_tried": channels_tried,
        "source_confidence": confidence,
        "deal_stage": "Sourced" if enriched else "Lost",
        "total_estimated_cost_usd": new_cost,
        "error_log": error_log,
    }