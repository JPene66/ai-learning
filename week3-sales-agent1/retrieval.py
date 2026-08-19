"""
retrieval.py
Implements Week 3, Skill 2 (Agentic Retrieval / Advanced RAG) as a set of
standalone, reusable functions - the same "one file per concept" spirit as
Week 2's pipeline files.

Concepts covered here:
  1. Query planning & decomposition   -> plan_source_queries()
  2. Multi-source retrieval routing    -> route_query()
  3. On-demand / dynamic retrieval      -> should_fetch_more()
  4. Self-querying retriever             -> extract_deal_filters()
  5. Iterative retrieve-refine loop       -> retrieve_refine_objection_response()
  8. Corrective RAG (CRAG) & fallback      -> score_relevance() + needs_fallback()

(Concept 6, GraphRAG, lives in knowledge_base.find_related(); Concept 7,
tool-augmented retrieval, is demonstrated across tools.py as a whole.)
"""

from helpers import ask_json, ask
from memory import query_long_term_memory
from knowledge_base import CASE_STUDIES, PRICING_FAQ


# --- Concept 1: Query Planning & Decomposition ---

def plan_source_queries(goal: str) -> list:
    """
    Breaks a high-level sourcing goal ('Find fintech leads under 200
    employees') into 2-5 concrete sub-queries across different channels,
    instead of running one vague search.
    """
    prompt = f"""Break this lead-sourcing goal into 2-4 concrete search sub-queries,
each targeting a different angle (industry directories, recent funding news,
referral/lookalike accounts, etc).

Goal: {goal}

Respond ONLY as JSON: {{"sub_queries": [list of short search query strings]}}"""
    result = ask_json(prompt)
    return result.get("sub_queries", [goal])


# --- Concept 2: Multi-Source Retrieval Routing ---

def route_query(query: str) -> str:
    """
    Classifies a query and returns which source should answer it - the
    routing layer described in the study guide ('financial data' -> SQL DB,
    'company policies' -> internal PDFs, 'latest news' -> web search).
    Apex's 3 sources: 'web' (new leads/news), 'crm' (existing accounts),
    'knowledge_base' (case studies, pricing, past deals).
    """
    q = query.lower()
    if any(k in q for k in ("price", "pricing", "discount", "cost", "contract")):
        return "knowledge_base"
    if any(k in q for k in ("existing", "already", "duplicate", "crm", "account history")):
        return "crm"
    return "web"


# --- Concept 3: On-Demand / Dynamic Retrieval ---

def should_fetch_more(known_fields: dict, required_fields: list) -> list:
    """
    Returns only the fields still missing, so the Qualify agent fetches
    ADDITIONAL data only when something is actually unknown - not a full
    re-fetch every time. This is the 'do I already know this?' check from
    the study guide.
    """
    return [f for f in required_fields if not known_fields.get(f)]


# --- Concept 4: Self-Querying Retriever ---

def extract_deal_filters(natural_language_query: str) -> dict:
    """
    Translates a natural-language request ('similar fintech deals over
    $40k that we won') into a structured metadata filter dict that can be
    applied directly to the PAST_DEALS vector store's `where` clause.
    """
    prompt = f"""Extract structured filters from this request about past sales deals.
Possible fields: industry (string), outcome ("won" or "lost"),
min_deal_size_usd (number, or null if not mentioned).

Request: {natural_language_query}

Respond ONLY as JSON: {{"industry": string or null, "outcome": string or null, "min_deal_size_usd": number or null}}"""
    return ask_json(prompt)


def build_chroma_where(filters: dict) -> dict:
    """Converts extract_deal_filters() output into a Chroma-compatible `where` clause."""
    clauses = []
    if filters.get("industry"):
        clauses.append({"industry": filters["industry"]})
    if filters.get("outcome"):
        clauses.append({"outcome": filters["outcome"]})
    if filters.get("min_deal_size_usd"):
        clauses.append({"deal_size_usd": {"$gte": filters["min_deal_size_usd"]}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# --- Concept 8: Corrective RAG (CRAG) & Fallback ---

def score_relevance(query: str, retrieved_text: str) -> float:
    """A lightweight relevance scorer (0-1) - the 'grade retrieved documents
    against the query' step CRAG needs to decide whether to trust a result."""
    result = ask_json(
        f"Query: {query}\nRetrieved text: {retrieved_text}\n\n"
        'Score 0.0-1.0 how relevant the retrieved text is to the query. '
        'Respond ONLY as JSON: {"relevance": float}'
    )
    return float(result.get("relevance", 0.0))


def needs_fallback(avg_relevance: float, threshold: float = 0.5) -> bool:
    return avg_relevance < threshold


# --- Concept 5: Iterative Retrieve-Refine Loop (used for objection handling) ---

def retrieve_refine_objection_response(objection_text: str, kb_collection, max_iterations: int = 2):
    """
    Retrieves supporting material for an objection, drafts a response,
    critiques its own draft for completeness, and retrieves again if gaps
    remain - stopping at max_iterations either way (loop guard).
    Returns (final_response, iterations_used, sources_used).
    """
    sources_used = []
    draft = None
    i = 0

    for i in range(max_iterations):
        matches = query_long_term_memory(kb_collection, objection_text, n_results=2)
        sources_used.extend(m.get("id", f"match_{i}") for m in matches if "id" in m)
        context = "\n".join(m["text"] for m in matches) if matches else "No directly relevant past deal notes found."

        if draft is None:
            prompt = f"Objection: {objection_text}\nRelevant history:\n{context}\n\nDraft a short, confident response to this objection."
        else:
            prompt = f"Previous draft: {draft}\nObjection: {objection_text}\nAdditional context:\n{context}\n\nRefine the draft to address any remaining gaps."
        draft = ask(prompt, max_tokens=200)

        critique = ask_json(
            f"Objection: {objection_text}\nDraft response: {draft}\n\n"
            'Does this fully address the objection, or is something missing? '
            'Respond ONLY as JSON: {"complete": true/false, "missing": string}'
        )
        if critique.get("complete", True):
            break

    return draft, min(i + 1, max_iterations), sources_used


def get_pricing_answer(topic_query: str) -> str:
    """Simple lookup helper over PRICING_FAQ, used as a cheap first pass
    before reaching for the heavier retrieve-refine loop."""
    q = topic_query.lower()
    for entry in PRICING_FAQ:
        if entry["topic"].replace("_", " ") in q or any(w in q for w in entry["topic"].split("_")):
            return entry["answer"]
    return ""