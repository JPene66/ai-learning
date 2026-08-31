"""
contradiction.py
Compares retrieved chunks from different sources and flags factual
contradictions between them - the signature feature of Case File AI.
"""

from helpers import ask_json

CONTRADICTION_PROMPT = """You are a careful fact-checking investigator reviewing evidence from
multiple sources about the same case. Below are excerpts, each labeled with its source.

Find any factual contradictions between sources - places where two or more sources
disagree about a specific fact (a time, a cause, who was present, what happened, etc).
Do NOT invent contradictions that aren't really there, and do NOT flag two sources as
contradicting just because one has more detail than the other - only flag genuine
disagreement about the same fact.

Respond ONLY as JSON in this exact shape:
{{
  "contradictions": [
    {{
      "topic": "short label for what the sources disagree about",
      "claim_a": "the first claim",
      "source_a": "which source said it",
      "claim_b": "the conflicting claim",
      "source_b": "which source said it",
      "severity": "Low|Medium|High"
    }}
  ]
}}
If there are no real contradictions, return {{"contradictions": []}}.

EVIDENCE:
{evidence_block}
"""


def format_evidence_block(retrieved_chunks):
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def find_contradictions(retrieved_chunks):
    """
    retrieved_chunks: list of dicts from vectorstore.hybrid_search(),
    each with at least 'text' and 'source' keys.
    Returns a list of contradiction dicts (empty list if none found).
    """
    if len(retrieved_chunks) < 2:
        return []  # can't contradict with only one piece of evidence

    evidence_block = format_evidence_block(retrieved_chunks)
    prompt = CONTRADICTION_PROMPT.format(evidence_block=evidence_block)
    result = ask_json(prompt)

    contradictions = result.get("contradictions", [])
    # Defensive defaults - never let one malformed entry break the UI
    for c in contradictions:
        c.setdefault("topic", "Unspecified")
        c.setdefault("claim_a", "")
        c.setdefault("source_a", "unknown")
        c.setdefault("claim_b", "")
        c.setdefault("source_b", "unknown")
        c.setdefault("severity", "Low")

    return contradictions