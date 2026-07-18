"""
Rubric-based evaluation using an LLM-as-judge (Concept 13).
This is the evaluation engine behind The Persuasion Arena's live scoring -
it turns "did that reply sound good?" into a real, tracked, structured score.
"""

from helpers import ask_json

RUBRIC_PROMPT = """You are a strict, consistent judge scoring a negotiation reply.
Score the REPLY on each criterion from 1 (very poor) to 5 (excellent):
- logic: Is the reasoning coherent and relevant to the offer?
- evidence: Does it reference specific facts, numbers, or concrete points (not vague filler)?
- clarity: Is it easy to understand and free of rambling?
- persuasiveness: How likely is this reply to move the negotiation forward in the persona's favor?

Respond ONLY as JSON in this exact shape:
{{"logic": int, "evidence": int, "clarity": int, "persuasiveness": int, "rationale": string}}

OFFER: {offer}
PERSONA: {persona_label}
REPLY: {reply}
"""


def score_reply(offer, persona_label, reply):
    prompt = RUBRIC_PROMPT.format(offer=offer, persona_label=persona_label, reply=reply)
    result = ask_json(prompt)

    # Defensive defaults - never let a single malformed judge response crash the app
    for key in ("logic", "evidence", "clarity", "persuasiveness"):
        result.setdefault(key, 0)
    result.setdefault("rationale", "")

    result["average"] = round(
        (result["logic"] + result["evidence"] + result["clarity"] + result["persuasiveness"]) / 4, 2
    )
    return result