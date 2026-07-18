"""
Core orchestration for a single round of The Persuasion Arena.
This is where prompt chaining (Concept 8) happens implicitly: the opponent's
reply is generated first, then fed into the scoring step as its input.
"""

from helpers import ask_with_system
from persona import get_system_prompt, PERSONAS
from scoring import score_reply


def run_round(persona_key, version, offer):
    """Runs one persona against one offer: generate a reply, then score it."""
    system_prompt = get_system_prompt(persona_key, version)
    reply = ask_with_system(system_prompt, offer)
    score = score_reply(offer, PERSONAS[persona_key]["label"], reply)

    return {
        "persona_key": persona_key,
        "persona_label": PERSONAS[persona_key]["label"],
        "version": version,
        "offer": offer,
        "reply": reply,
        "score": score,
    }


def run_multi_persona_round(persona_keys, version, offer):
    """Runs the same offer against several personas at once, for side-by-side comparison."""
    return [run_round(key, version, offer) for key in persona_keys]