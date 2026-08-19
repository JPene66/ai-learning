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
from guardrails import is_tool_allowed


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