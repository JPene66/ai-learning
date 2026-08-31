"""
tools.py
Apex's tools (Week 3, Skill 1, Concept 2: Tool / Function Calling).

Every tool here is SIMULATED - deterministic, free, and needs no external
API key beyond OpenAI's. This is deliberate: the point of this project is
to demonstrate agent architecture, not to depend on a paid web-search or
CRM API. Swapping any tool for a real one later means editing only the
function body - every schema and calling convention stays identical.

Each tool has:
  - a JSON schema (name, description, parameters) - Concept 2's requirement
  - the actual function
  - registration in TOOL_REGISTRY, used by execute_tool() for validation

execute_tool() is the "tool executor that validates arguments, calls the
function, and handles exceptions" the study guide's hands-on asks for.
"""

import random
import time
import json
from guardrails import is_tool_allowed

try:
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    _DDG_AVAILABLE = False


# --- Simulated data pools (deterministic-ish, seeded by input for reproducibility) ---

_INDUSTRY_COMPANY_POOL = {
    "fintech": ["Ledgerly", "PayNova", "Vaultwise", "Fintrace", "Coincircuit"],
    "healthtech": ["MedBridge", "Carelytics", "Vitalstack", "Clinovate"],
    "logistics": ["Routewise", "Freightnest", "Cargotrail", "Shipcore"],
    "retail": ["Shelfsense", "Stockloop", "Retailix", "Cartify"],
}


def _seeded_random(seed_text: str) -> random.Random:
    return random.Random(hash(seed_text) % (2**31))


# --- Tool 1: search_web_leads ---

TOOL_SCHEMA_SEARCH_WEB_LEADS = {
    "name": "search_web_leads",
    "description": "Searches the web for candidate companies matching an industry and size criteria.",
    "parameters": {
        "type": "object",
        "properties": {
            "industry": {"type": "string", "description": "Target industry, e.g. 'fintech'"},
            "max_results": {"type": "integer", "description": "Max number of leads to return"},
        },
        "required": ["industry"],
    },
}


def search_web_leads(industry: str, max_results: int = 3):
    industry_key = industry.lower().strip()
    pool = _INDUSTRY_COMPANY_POOL.get(industry_key, ["GenericCo", "SampleWorks", "DemoTech"])
    rng = _seeded_random(industry_key)
    chosen = rng.sample(pool, min(max_results, len(pool)))
    return [
        {"company": name, "industry": industry_key, "source": "web_search",
         "employee_estimate": rng.choice([40, 85, 120, 210, 350])}
        for name in chosen
    ]


# --- Tool 1b: search_real_leads (DuckDuckGo-powered, live) ---

TOOL_SCHEMA_SEARCH_REAL_LEADS = {
    "name": "search_real_leads",
    "description": (
        "Searches the web via DuckDuckGo for real companies in a specific country, "
        "region, and industry. Returns structured lead dicts with company name, "
        "description, and estimated employee count."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "industry": {"type": "string", "description": "Target sector, e.g. 'Banking'"},
            "country": {"type": "string", "description": "Country name, e.g. 'Nigeria'"},
            "region": {"type": "string", "description": "Region/city, e.g. 'Lagos'"},
            "max_results": {"type": "integer", "description": "Number of leads to return (1-5)"},
        },
        "required": ["industry", "country"],
    },
}


def search_real_leads(industry: str, country: str, region: str = "", max_results: int = 2):
    """
    Live lead search using DuckDuckGo.

    1. Builds a targeted query string like:
       "top banking companies in Lagos Nigeria"
    2. Fires a DuckDuckGo text search to get raw snippets (3x the target
       count to allow for bad/irrelevant hits that get filtered out).
    3. Uses a cheap GPT ask_json() call to parse the raw snippets into
       structured lead dicts: [{company, industry, country, region,
       website, description, employee_estimate}].
    4. Falls back to an empty list on any error so the caller's corrective
       fallback logic handles the empty case cleanly.
    """
    from helpers import ask_json  # local import to avoid circular at module level

    location_part = f"{region}, {country}" if region else country
    query = f"top {industry} companies in {location_part}"

    raw_snippets = []
    if _DDG_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max(max_results * 3, 6)))
            raw_snippets = [
                f"{r.get('title', '')} — {r.get('body', '')[:300]}"
                for r in results
                if r.get("title") or r.get("body")
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[search_real_leads] DuckDuckGo error: {exc}")
    else:
        print("[search_real_leads] duckduckgo-search not installed; returning empty.")
        return []

    if not raw_snippets:
        return []

    snippets_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(raw_snippets[:15]))
    parse_prompt = f"""From the search snippets below, extract up to {max_results} REAL companies
that operate in the {industry} sector in {location_part}.

For each company return:
  - company: the exact company name (string)
  - industry: "{industry}" (string)
  - country: "{country}" (string)
  - region: "{region}" (string)
  - description: a 1-sentence description of what they do (string)
  - employee_estimate: rough headcount as an integer (use 50 if unknown)
  - source: "duckduckgo" (string)

Ignore generic directory sites, news articles, or snippets that don't name a specific company.

Search snippets:
{snippets_text}

Respond ONLY as valid JSON: {{"leads": [list of company objects as described above]}}"""

    try:
        parsed = ask_json(parse_prompt)
        leads = parsed.get("leads", [])
        # Sanitise: make sure every required key exists
        clean = []
        for lead in leads[:max_results]:
            if isinstance(lead, dict) and lead.get("company"):
                lead.setdefault("industry", industry)
                lead.setdefault("country", country)
                lead.setdefault("region", region)
                lead.setdefault("source", "duckduckgo")
                lead.setdefault("employee_estimate", 50)
                lead.setdefault("description", "")
                clean.append(lead)
        return clean
    except Exception as exc:  # noqa: BLE001
        print(f"[search_real_leads] GPT parse error: {exc}")
        return []


# --- Tool 2: enrich_company ---

TOOL_SCHEMA_ENRICH_COMPANY = {
    "name": "enrich_company",
    "description": "Fetches firmographic detail (funding, headcount, tech stack) for a named company.",
    "parameters": {
        "type": "object",
        "properties": {"company": {"type": "string"}},
        "required": ["company"],
    },
}


def enrich_company(company: str):
    rng = _seeded_random(company.lower())
    return {
        "company": company,
        "employee_count": rng.choice([35, 60, 90, 150, 220, 400]),
        "estimated_budget_usd": rng.choice([8000, 15000, 30000, 50000, 75000]),
        "recently_funded": rng.choice([True, False]),
        "known_pain_point": rng.choice([
            "manual compliance reporting", "slow onboarding", "inventory forecasting",
            "dispatch scheduling", "data reconciliation",
        ]),
    }


# --- Tool 3: check_crm_duplicate ---

TOOL_SCHEMA_CHECK_CRM_DUPLICATE = {
    "name": "check_crm_duplicate",
    "description": "Checks whether a company already exists as a lead or account in the CRM.",
    "parameters": {
        "type": "object",
        "properties": {"company": {"type": "string"}},
        "required": ["company"],
    },
}

_CRM_SEEN = {"Ledgerly", "Carelytics"}  # a couple of companies simulated as "already known"


def check_crm_duplicate(company: str):
    return {"company": company, "is_duplicate": company in _CRM_SEEN}


# --- Tool 4: book_meeting ---

TOOL_SCHEMA_BOOK_MEETING = {
    "name": "book_meeting",
    "description": "Books a meeting slot for a qualified, engaged lead.",
    "parameters": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "requested_day": {"type": "string", "description": "e.g. 'next Tuesday'"},
        },
        "required": ["company"],
    },
}


def book_meeting(company: str, requested_day: str = "next available slot"):
    return {"company": company, "booked": True, "slot": requested_day, "confirmation_id": f"MTG-{abs(hash(company)) % 10000}"}


# --- Tool executor: validates arguments, calls the function, handles exceptions ---

TOOL_REGISTRY = {
    "search_web_leads": (search_web_leads, TOOL_SCHEMA_SEARCH_WEB_LEADS),
    "search_real_leads": (search_real_leads, TOOL_SCHEMA_SEARCH_REAL_LEADS),
    "enrich_company": (enrich_company, TOOL_SCHEMA_ENRICH_COMPANY),
    "check_crm_duplicate": (check_crm_duplicate, TOOL_SCHEMA_CHECK_CRM_DUPLICATE),
    "book_meeting": (book_meeting, TOOL_SCHEMA_BOOK_MEETING),
}


def execute_tool(tool_name: str, arguments: dict, max_retries: int = 1):
    """
    Validates the tool is allow-listed and required args are present, then
    calls it - catching and reporting exceptions instead of crashing the
    whole agent run. Returns (success: bool, result_or_error).
    """
    if not is_tool_allowed(tool_name):
        return False, f"Tool '{tool_name}' is not on the allow-list."

    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool: '{tool_name}'"

    func, schema = TOOL_REGISTRY[tool_name]
    required = schema["parameters"].get("required", [])
    missing = [r for r in required if r not in arguments]
    if missing:
        return False, f"Missing required argument(s) for {tool_name}: {missing}"

    attempts = 0
    last_error = None
    while attempts <= max_retries:
        try:
            result = func(**arguments)
            return True, result
        except Exception as e:  # noqa: BLE001 - intentional broad catch for tool-call robustness
            last_error = str(e)
            attempts += 1
            time.sleep(0)  # placeholder for real backoff if a real API were involved

    return False, f"Tool '{tool_name}' failed after {attempts} attempt(s): {last_error}"