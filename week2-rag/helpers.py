"""
Self-contained LLM + embedding utilities for Case File AI.
Same pattern as Project 1's helpers.py - this project lives in its own
repo and doesn't reach back into other folders.

Every LLM and embedding call in the whole pipeline goes through this file,
so there is exactly ONE place that knows about the model, the API key, and
the request format. If you ever swap providers, this is the only file that
has to change.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Two lookups on purpose: the first searches upward from this file (so the CLI
# scripts in this folder work), the second checks the directory the process was
# launched from (so `streamlit run app.py` inside Project-2 finds Project-2/.env).
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

# A placeholder key lets the module import (and the unit tests run) without a
# real key configured - the call itself will fail loudly, which is what we want.
API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder-for-tests")

client = OpenAI(api_key=API_KEY)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class MissingAPIKeyError(RuntimeError):
    """Raised with a friendly message instead of a raw 401 from the API."""


def _check_key():
    if API_KEY == "sk-placeholder-for-tests" or not API_KEY.strip():
        raise MissingAPIKeyError(
            "No OPENAI_API_KEY found. Copy .env.example to .env and put your "
            "key in it, then restart the app."
        )


def ask(prompt, model=DEFAULT_MODEL, max_tokens=800, temperature=0.2):
    """Plain text completion. Low temperature by default because every use in
    this project is grounded, factual work - we want the model sticking to the
    evidence, not being creative."""
    _check_key()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def ask_json(prompt, system_prompt=None, model=DEFAULT_MODEL, max_tokens=900):
    """
    Same as ask(), but forces the model to reply with valid JSON so the result
    can be used as data instead of prose. Used by re-ranking, contradiction
    detection and evaluation - every step where we need a structure back.
    """
    _check_key()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        print("Warning: model did not return valid JSON:", raw)
        return {}


def embed(text, model=EMBEDDING_MODEL):
    """Embed one string. Used for the query side (query_embed.py)."""
    _check_key()
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def embed_many(texts, model=EMBEDDING_MODEL, batch_size=64):
    """
    Embed a whole list of strings.

    The OpenAI embeddings endpoint accepts a list as input, so embedding 200
    chunks can be 4 requests instead of 200. Ingesting a full PDF one chunk at
    a time is the single slowest thing this project does, and batching is what
    fixes it. Results come back in the same order they were sent.
    """
    _check_key()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        # sort by index so we never depend on the API preserving order for us
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return vectors
