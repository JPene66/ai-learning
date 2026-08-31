"""
Self-contained LLM + embedding utilities for Case File AI.
Same pattern as Project 1's helpers.py - this project lives in its own
repo and doesn't reach back into other folders.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-placeholder-for-tests"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def ask(prompt, model=DEFAULT_MODEL, max_tokens=500, temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def ask_json(prompt, system_prompt=None, model=DEFAULT_MODEL, max_tokens=600):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: model did not return valid JSON:", raw)
        return {}


def embed(text, model=EMBEDDING_MODEL):
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding