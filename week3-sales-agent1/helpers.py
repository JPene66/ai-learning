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