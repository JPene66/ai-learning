"""
helpers.py
Shared OpenAI client + reusable functions for Apex, the autonomous sales agent.

Cost/latency tradeoff design (Week 3, Skill 1, Concept 10): two model tiers
are defined - a cheap/fast ROUTING_MODEL for scoring, classification, and
routing decisions (high volume, low complexity), and a stronger
GENERATION_MODEL for anything customer-facing (outreach copy, objection
handling) where quality matters more than cost. Both default to the same
model today, but the separation is deliberate so swapping just the routing
model later is a one-line change, not a refactor.

Embeddings use Chroma's built-in local ONNX MiniLM model (no OpenAI call,
no PyTorch/CUDA dependency) - consistent with the program's "don't spend on
embeddings" constraint.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder-for-tests"))

ROUTING_MODEL = os.getenv("ROUTING_MODEL", "gpt-4o-mini")      # scoring, classification, routing
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")  # customer-facing copy
WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "gpt-4o")  


def ask(prompt, model=GENERATION_MODEL, max_tokens=500, temperature=0.5, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature, messages=messages,
    )
    return response.choices[0].message.content


def ask_json(prompt, system_prompt=None, model=ROUTING_MODEL, max_tokens=700, temperature=0.2):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: model did not return valid JSON:", raw)
        return {}

def web_search(query: str, user_location: dict = None, allowed_domains: list = None) -> str:
    """
    Real, live web search via OpenAI's hosted web_search tool (Responses
    API) - this replaces the old hardcoded company pools entirely. The
    model both searches and reads the results, returning prose grounded in
    live data (with citations attached internally). This is a genuinely
    different, and more expensive, kind of call than ask()/ask_json() - it
    costs a flat per-call fee on top of normal token costs, so callers
    should be deliberate about how often they invoke it (see
    guardrails.MAX_WEB_SEARCHES_PER_SOURCE_RUN).
 
    user_location: optional dict like {"country": "NG", "city": "Lagos"} -
    nudges results toward that region without guaranteeing exact filtering,
    since arbitrary cities aren't a hard geographic filter on OpenAI's side.
    Passing the location directly in `query` text is usually just as
    effective and is what search_web_leads() actually relies on.
    """
    tool_config = {"type": "web_search"}
    if user_location:
        tool_config["user_location"] = {"type": "approximate", **user_location}
    if allowed_domains:
        tool_config["filters"] = {"allowed_domains": allowed_domains}
 
    response = client.responses.create(
        model=WEB_SEARCH_MODEL,
        tools=[tool_config],
        input=query,
    )
    return response.output_text