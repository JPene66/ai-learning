"""
qualify_agent.py
The Qualify node: "Scores and segments leads using BANT+MEDICC criteria."

BANT   = Budget, Authority, Need, Timeline
MEDICC = Metrics, Economic buyer, Decision criteria, Identify pain,
         Champion, Competition
(Note: this is a 6-factor variant. If you intended the standard 6-factor
MEDDIC - Metrics, Economic buyer, Decision criteria, Decision process,
Identify pain, Champion - swap "Competition" for "Decision process" below;
both are one-line changes in RUBRIC_CRITERIA.)

Demonstrates:
  - On-demand / dynamic retrieval: only fetches more data for fields
    that are actually still missing (retrieval.should_fetch_more)
  - Self-querying retriever: turns "similar deals we've won in this
    industry" into a structured filter against the long-term memory
    vector store (retrieval.extract_deal_filters + build_chroma_where)
  - Cost guardrail tracking
"""

from helpers import ask_json
from retrieval import should_fetch_more, extract_deal_filters, build_chroma_where
from memory import get_long_term_memory, seed_long_term_memory, query_long_term_memory
from tools import execute_tool
from guardrails import record_cost

REQUIRED_FIELDS = ["estimated_budget_usd", "employee_count", "known_pain_point"]

RUBRIC_CRITERIA = [
    "budget", "authority", "need", "timeline",
    "metrics", "economic_buyer", "decision_criteria",
    "identify_pain", "champion", "competition",
]

HOT_THRESHOLD = 75
WARM_THRESHOLD = 50
COLD_THRESHOLD = 25


def _fetch_missing_fields(lead: dict) -> dict:
    missing = should_fetch_more(lead, REQUIRED_FIELDS)
    if not missing:
        return lead
    ok, result = execute_tool("enrich_company", {"company": lead.get("company", "unknown")})
    return {**lead, **result} if ok else lead


def _find_similar_past_deals(lead: dict, collection) -> list:
    nl_query = f"similar {lead.get('industry', 'unknown')} deals we've won"
    filters = extract_deal_filters(nl_query)
    where = build_chroma_where(filters)
    return query_long_term_memory(collection, nl_query, n_results=3, where=where)


def _score_bant_medicc(lead: dict, similar_deals: list) -> dict:
    deals_context = "\n".join(f"- {d['text']}" for d in similar_deals) or "No closely similar past deals found."
    prompt = f"""Score this lead on BANT+MEDICC criteria, each from 1 (very weak) to 5 (very strong),
based on the lead data and similar past deal history below.

LEAD DATA:
{lead}

SIMILAR PAST DEALS (for calibration):
{deals_context}

Score these 10 criteria: budget, authority, need, timeline, metrics, economic_buyer,
decision_criteria, identify_pain, champion, competition.

Respond ONLY as JSON in this shape:
{{"scores": {{"budget": int, "authority": int, "need": int, "timeline": int, "metrics": int,
"economic_buyer": int, "decision_criteria": int, "identify_pain": int, "champion": int,
"competition": int}}, "rationale": string}}"""
    result = ask_json(prompt)
    scores = result.get("scores", {})
    for c in RUBRIC_CRITERIA:
        scores.setdefault(c, 3)
    return {"scores": scores, "rationale": result.get("rationale", "")}


def _aggregate_score(scores: dict) -> float:
    total = sum(scores.get(c, 0) for c in RUBRIC_CRITERIA)
    return round((total / (len(RUBRIC_CRITERIA) * 5)) * 100, 1)


def _segment(score: float) -> str:
    if score >= HOT_THRESHOLD:
        return "Hot"
    if score >= WARM_THRESHOLD:
        return "Warm"
    if score >= COLD_THRESHOLD:
        return "Cold"
    return "Disqualified"


def qualify_node(state: dict) -> dict:
    lead = _fetch_missing_fields(state.get("lead", {}))

    collection = get_long_term_memory()
    seed_long_term_memory(collection)
    similar_deals = _find_similar_past_deals(lead, collection)

    scoring_result = _score_bant_medicc(lead, similar_deals)
    score = _aggregate_score(scoring_result["scores"])
    segment = _segment(score)

    new_cost = record_cost(state, 0.015)

    return {
        "lead": lead,
        "bant_medicc": scoring_result,
        "qualification_score": score,
        "segment": segment,
        "deal_stage": "Qualified" if segment in ("Hot", "Warm") else "Disqualified",
        "total_estimated_cost_usd": new_cost,
    }