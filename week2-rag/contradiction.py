"""
contradiction.py
(QUERY lane): the signature feature of Case File AI - compare the retrieved
evidence ACROSS different sources and flag places where they genuinely
disagree about a fact.

Input:  the question + the re-ranked chunks (from rerank.py)
Output: a list of contradiction records, each naming both sides and how
        serious the disagreement is

Why this is its own step: a normal RAG answer quietly averages over its
sources. If your evidence says "the meeting was on the 3rd" and "the meeting
was on the 9th", a plain RAG system picks one and sounds equally confident
either way. Detecting that disagreement - and surfacing it - is the whole
point of a case-file tool.
"""

from helpers import ask_json

# Only these three severities are ever shown to the user. Anything else the
# model invents ("critical", "minor", "moderate"...) gets mapped or defaulted.
VALID_SEVERITIES = ("high", "medium", "low")

SEVERITY_ALIASES = {
    "critical": "high",
    "severe": "high",
    "major": "high",
    "moderate": "medium",
    "medium-high": "high",
    "minor": "low",
    "trivial": "low",
}

DEFAULT_SEVERITY = "medium"

PREVIEW_CHARS = 1200


def _normalize_severity(raw):
    """Coerces whatever the model said into one of our three severities."""
    if not isinstance(raw, str):
        return DEFAULT_SEVERITY
    value = raw.strip().lower()
    if value in VALID_SEVERITIES:
        return value
    return SEVERITY_ALIASES.get(value, DEFAULT_SEVERITY)


def _source_name(index, chunks):
    """
    Maps a 1-based [Source N] number back to the real filename/URL.
    Falls back to a readable placeholder rather than crashing if the model
    cites a source number that doesn't exist.
    """
    try:
        i = int(index)
    except (TypeError, ValueError):
        return "unknown source"
    if 1 <= i <= len(chunks):
        return chunks[i - 1]["source"]
    return f"unknown source ({index})"


def normalize_contradictions(raw_result, chunks):
    """
    Turns the model's raw JSON into clean, displayable records - applying a
    default for every field the model might omit, and dropping anything that
    isn't actually a usable contradiction.

    Kept as a separate, pure function (no API calls) precisely so it can be
    unit-tested against the malformed responses models really produce.
    """
    if not isinstance(raw_result, dict):
        return []

    items = raw_result.get("contradictions", [])
    if not isinstance(items, list):
        return []

    contradictions = []
    for item in items:
        if not isinstance(item, dict):
            continue

        claim_a = (item.get("claim_a") or "").strip()
        claim_b = (item.get("claim_b") or "").strip()

        # A "contradiction" with only one side isn't a contradiction.
        if not claim_a or not claim_b:
            continue

        source_a = _source_name(item.get("source_a"), chunks)
        source_b = _source_name(item.get("source_b"), chunks)

        # Two claims from the same document aren't a cross-source conflict -
        # that's usually the model re-reading one passage twice.
        if source_a == source_b:
            continue

        contradictions.append({
            "claim_a": claim_a,
            "source_a": source_a,
            "claim_b": claim_b,
            "source_b": source_b,
            "severity": _normalize_severity(item.get("severity")),
            "explanation": (item.get("explanation") or "").strip()
                           or "The two sources state different things about the same fact.",
        })

    return contradictions


def detect_contradictions(question, chunks, ask_json_fn=None):
    """
    Asks the LLM to compare the retrieved evidence for real factual conflicts.

    Returns [] without spending an API call when there's nothing to compare -
    a contradiction needs at least two DIFFERENT sources by definition.

    ask_json_fn lets tests inject a fake LLM; production uses helpers.ask_json.
    """
    if not chunks:
        return []

    distinct_sources = {c["source"] for c in chunks}
    if len(distinct_sources) < 2:
        return []

    ask_json_fn = ask_json_fn or ask_json

    numbered = "\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['text'][:PREVIEW_CHARS]}"
        for i, c in enumerate(chunks, start=1)
    )

    prompt = f"""You are auditing evidence for factual disagreements.

QUESTION UNDER INVESTIGATION: {question}

EVIDENCE:
{numbered}

Find places where two DIFFERENT sources make claims that cannot both be true -
different dates, numbers, names, outcomes, or directly opposing statements
about the same thing.

Do NOT report:
- Two sources simply covering different topics (that is not a contradiction).
- One source having more detail than another.
- Differences in wording, tone, or emphasis.
- Two passages from the same source.

Rate severity:
- "high"   = the two claims are directly incompatible on a central fact
- "medium" = they conflict on a supporting detail
- "low"    = they are in mild tension and might be reconcilable

Respond ONLY as JSON:
{{"contradictions": [
  {{"claim_a": "what one source says",
    "source_a": <source number>,
    "claim_b": "what the other source says",
    "source_b": <source number>,
    "severity": "high" | "medium" | "low",
    "explanation": "one sentence on why these cannot both be true"}}
]}}

If the sources do not actually contradict each other, return {{"contradictions": []}}."""

    try:
        raw = ask_json_fn(prompt)
    except Exception as e:
        # A failed audit must not take down the answer - report "none found"
        # and let the confidence score reflect the missing check.
        print(f"Contradiction detection failed ({e}) - continuing without it.")
        return []

    return normalize_contradictions(raw, chunks)


def summarize_contradictions(contradictions):
    """A one-line headline for the UI, e.g. '2 conflicts found (1 high)'."""
    if not contradictions:
        return "No conflicts detected between sources."

    high = sum(1 for c in contradictions if c["severity"] == "high")
    total = len(contradictions)
    noun = "conflict" if total == 1 else "conflicts"
    headline = f"{total} {noun} found between sources"
    if high:
        headline += f" ({high} high severity)"
    return headline + "."


if __name__ == "__main__":
    from search import hybrid_search
    from rerank import rerank
    from storing import get_collection

    collection = get_collection()
    question = "What date did the incident occur?"

    candidates = hybrid_search(collection, question, n_results=8)
    top_chunks = rerank(question, candidates, keep_top=5)

    found = detect_contradictions(question, top_chunks)
    print(summarize_contradictions(found))
    for c in found:
        print(f"\n[{c['severity'].upper()}] {c['explanation']}")
        print(f"  {c['source_a']}: {c['claim_a']}")
        print(f"  {c['source_b']}: {c['claim_b']}")
