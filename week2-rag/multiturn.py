"""
multiturn.py
(QUERY lane): make the assistant conversational.

The problem this solves is specific and easy to miss. A user asks:

    "What did the witness say about the car?"
    "And what time was that?"

Embedding "And what time was that?" on its own retrieves nothing useful - the
words "witness" and "car" aren't in it, so the vector is about nothing. The fix
is to REWRITE the follow-up into a standalone question using the conversation
history, and retrieve with the rewritten version.

So there are two different jobs here, and keeping them separate matters:
  1. Rewriting the follow-up for RETRIEVAL (this file)
  2. Passing history to the LLM for GENERATION (context_assembly.build_prompt)
"""

from helpers import ask

# Only the last few turns are used. Older turns rarely help resolve a pronoun
# and just cost tokens on every question.
HISTORY_TURNS_FOR_REWRITE = 6


def add_turn(history, role, content):
    """Appends one turn and returns the history - a tiny helper so app.py and
    the CLI both build history in exactly the same shape."""
    history = list(history or [])
    history.append({"role": role, "content": content})
    return history


def needs_rewrite(question, history):
    """
    Cheap check for whether a rewrite is even worth an API call.

    With no history there is nothing to resolve against. A long question is
    usually already self-contained. Everything else gets rewritten - it's
    cheaper to rewrite unnecessarily than to retrieve on a broken query.
    """
    if not history:
        return False
    return len(question.split()) <= 12


def rewrite_followup(question, history, ask_fn=None):
    """
    Turns a context-dependent follow-up into a standalone question.

    Returns the ORIGINAL question unchanged whenever rewriting isn't needed or
    fails - a slightly worse retrieval is much better than a crash or a
    rewritten question that lost the user's actual intent.

    ask_fn lets tests inject a fake LLM instead of calling the real API.
    """
    if not needs_rewrite(question, history):
        return question

    ask_fn = ask_fn or ask

    recent = history[-HISTORY_TURNS_FOR_REWRITE:]
    transcript = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in recent
        if turn.get("content")
    )

    prompt = f"""Here is a conversation:

{transcript}

The user now asks: "{question}"

Rewrite that into a single standalone question that makes sense with no
conversation history - replace pronouns and vague references ("it", "that",
"they", "the same") with the specific things they refer to.

Change nothing else. Do not answer the question. Do not add information that
is not implied by the conversation. If the question already stands alone,
repeat it back unchanged.

Standalone question:"""

    try:
        rewritten = (ask_fn(prompt) or "").strip().strip('"')
    except Exception as e:
        print(f"Follow-up rewriting failed ({e}) - using the original question.")
        return question

    # Guard against the model returning an explanation, an empty string, or a
    # whole paragraph instead of a question.
    if not rewritten or len(rewritten) > 300:
        return question

    return rewritten


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "What did the witness say about the car?"},
        {"role": "assistant", "content": "The witness described a blue sedan leaving at speed [Source 1]."},
    ]
    print(rewrite_followup("And what time was that?", history))
