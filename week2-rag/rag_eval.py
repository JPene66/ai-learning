"""
rag_eval.py
(QUERY lane, optional): grade the pipeline's own output.

Three standard RAG metrics, each answering a different question:

  faithfulness - does the answer only say things the evidence actually supports?
                 (catches hallucination)
  relevance    - does the answer address what the user asked?
                 (catches a well-grounded answer to the wrong question)
  groundedness - are the claims traceable to specific, cited sources?
                 (catches confident prose with no citations behind it)

The judge is a separate LLM call that sees the evidence and the answer but NOT
the reasoning that produced it, so it grades the output on its own merits.
This is "LLM-as-judge" - useful and cheap, but it is an estimate, not truth.
"""

from helpers import ask_json

METRICS = ("faithfulness", "relevance", "groundedness")

PREVIEW_CHARS = 1200


def _clamp_score(value):
    """Models return 0.9, '0.9', 90, or nonsense. Normalize all of it to 0-1."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1:          # a 0-100 or 0-10 style score
        score = score / 100 if score > 10 else score / 10
    return round(max(0.0, min(1.0, score)), 2)


def normalize_evaluation(raw_result):
    """
    Cleans the judge's JSON into a predictable shape, with None for any metric
    it failed to return. Pure function, so it's unit-testable without an API.
    """
    if not isinstance(raw_result, dict):
        raw_result = {}

    scores = {metric: _clamp_score(raw_result.get(metric)) for metric in METRICS}

    reported = [s for s in scores.values() if s is not None]
    scores["overall"] = round(sum(reported) / len(reported), 2) if reported else None
    scores["notes"] = (raw_result.get("notes") or "").strip()
    return scores


def evaluate_answer(question, answer, chunks, ask_json_fn=None):
    """
    Scores one answer against the evidence it was built from.

    ask_json_fn lets tests inject a fake judge instead of calling the API.
    """
    if not answer or not chunks:
        return normalize_evaluation({})

    ask_json_fn = ask_json_fn or ask_json

    evidence = "\n\n".join(
        f"[Source {i}: {c['source']}]\n{c['text'][:PREVIEW_CHARS]}"
        for i, c in enumerate(chunks, start=1)
    )

    prompt = f"""Grade this answer produced by a retrieval-augmented system.

QUESTION: {question}

EVIDENCE THE ANSWER WAS GIVEN:
{evidence}

ANSWER PRODUCED:
{answer}

Score each metric from 0.0 to 1.0:
- faithfulness: every claim in the answer is supported by the evidence, with
  nothing added from outside it. Score low if anything was invented.
- relevance: the answer addresses the question that was actually asked.
- groundedness: claims carry [Source N] citations that match the evidence and
  are attributed to the right source.

An answer that correctly says the evidence does not cover the question should
score HIGH on faithfulness.

Respond ONLY as JSON:
{{"faithfulness": 0.0, "relevance": 0.0, "groundedness": 0.0,
  "notes": "one sentence on the weakest metric"}}"""

    try:
        raw = ask_json_fn(prompt)
    except Exception as e:
        print(f"Evaluation failed ({e}) - skipping.")
        return normalize_evaluation({})

    return normalize_evaluation(raw)


if __name__ == "__main__":
    from pipeline import answer_question
    from storing import get_collection

    collection = get_collection()
    result = answer_question(
        get_collection(), "What certifications are found in the resume?", evaluate=True
    )
    print(result["evaluation"])
