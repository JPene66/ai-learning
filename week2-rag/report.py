"""
report.py
(QUERY lane): turn everything the pipeline produced - the answer, the chunks
it was grounded in, and any contradictions found - into a final, explainable
result for the user.

The centrepiece is confidence scoring. Note that this is a HEURISTIC, not a
model output, and that's deliberate: asking an LLM "how confident are you?"
produces a number it cannot justify. Computing the score from things we can
actually observe (how strong the retrieval was, how many independent sources
agree, whether they contradict each other, whether the answer cited anything)
means every point of the score can be shown to the user and argued with.

This module is pure Python with no API calls, which is what makes the scoring
math directly unit-testable.
"""

# How much each contradiction subtracts from the score. A direct conflict on a
# central fact should meaningfully shake confidence; mild tension shouldn't.
SEVERITY_PENALTY = {"high": 0.35, "medium": 0.20, "low": 0.08}

# Cap on the total contradiction penalty, so ten trivial conflicts can't drive
# an otherwise well-evidenced answer to zero.
MAX_CONTRADICTION_PENALTY = 0.60

# Evidence strength is averaged over the best few chunks rather than all of
# them - a weak 5th result shouldn't drag down an answer that the top 3
# supported clearly.
TOP_CHUNKS_CONSIDERED = 3

# Raw retrieval scores are NOT percentages of correctness, and using them as
# one would quietly mis-calibrate the whole system. With text-embedding-3-small,
# unrelated text still scores around 0.1-0.2 cosine similarity, and a genuinely
# on-point passage lands around 0.5-0.6 - it very rarely approaches 1.0. Reported
# raw, a perfect retrieval would display as "50% confident", which is wrong.
#
# So the observed range is stretched onto a real 0-1 scale: at or below the
# floor is "no better than unrelated text", at or above the ceiling is "as good
# as this retriever gets". Retune these two numbers if you change embedding model.
RELEVANCE_FLOOR = 0.20
RELEVANCE_CEILING = 0.60

SINGLE_SOURCE_PENALTY = 0.10
PER_EXTRA_SOURCE_BONUS = 0.05
MAX_CORROBORATION_BONUS = 0.15

UNCITED_ANSWER_PENALTY = 0.10

# When the model correctly says "the evidence doesn't cover this", confidence
# in the ANSWER shouldn't be reported as low - but confidence that the case
# file ANSWERS the question genuinely is low. We cap it and say why.
REFUSAL_CAP = 0.20

REFUSAL_MARKERS = (
    "does not contain",
    "doesn't contain",
    "no evidence",
    "not covered",
    "does not cover",
    "cannot answer",
    "can't answer",
    "no information",
    "not mentioned",
    "insufficient evidence",
    "not enough information",
)


def _best_score(chunk):
    """
    How well this chunk actually matches the question, 0-1.

    Vector similarity is preferred over the blended hybrid score, and the
    distinction matters. The blended score exists to RANK candidates, and it
    deliberately rewards literal word overlap so exact terms aren't missed. But
    keyword overlap is a poor measure of QUALITY: a natural-language question
    ("what date did the collision happen?") shares few literal words with even
    a perfect answer passage, so the keyword component drags the blend down and
    would make good evidence look mediocre. Semantic similarity is the honest
    answer to "how relevant is this passage really?".
    """
    for key in ("vector_similarity", "blended_score", "keyword_score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    return 0.0


def _calibrate_relevance(raw_score):
    """Stretches a raw retrieval score onto a meaningful 0-1 scale - see the
    note on RELEVANCE_FLOOR above for why this is necessary."""
    span = RELEVANCE_CEILING - RELEVANCE_FLOOR
    calibrated = (raw_score - RELEVANCE_FLOOR) / span
    return round(max(0.0, min(1.0, calibrated)), 3)


def _looks_like_refusal(answer):
    """True when the answer is an honest 'the evidence doesn't say'."""
    if not answer:
        return False
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score_confidence(chunks, contradictions=None, answer=None):
    """
    Produces an explainable 0-1 confidence score.

    Returns:
        {
          "score":   0.0-1.0,
          "percent": 0-100,
          "label":   "High" | "Moderate" | "Low" | "Very low" | "No evidence",
          "factors": [{"label": ..., "effect": +/-float, "detail": ...}, ...]
        }

    'factors' is the important part - it's what the UI shows so a user can see
    exactly why the number is what it is.
    """
    contradictions = contradictions or []
    factors = []

    if not chunks:
        return {
            "score": 0.0,
            "percent": 0,
            "label": "No evidence",
            "factors": [{
                "label": "No evidence retrieved",
                "effect": 0.0,
                "detail": "Nothing in the case file matched this question.",
            }],
        }

    # 1. Evidence strength - how well the retrieved chunks actually matched.
    top_scores = sorted(
        (_calibrate_relevance(_best_score(c)) for c in chunks), reverse=True
    )[:TOP_CHUNKS_CONSIDERED]
    evidence_strength = sum(top_scores) / len(top_scores)
    score = evidence_strength
    factors.append({
        "label": "Evidence strength",
        "effect": round(evidence_strength, 3),
        "detail": f"Top {len(top_scores)} retrieved passages matched the question "
                  f"with an average relevance of {evidence_strength:.0%}.",
    })

    # 2. Corroboration - do independent sources back this up?
    distinct_sources = {c.get("source", "unknown") for c in chunks}
    source_count = len(distinct_sources)

    if source_count == 1:
        score -= SINGLE_SOURCE_PENALTY
        factors.append({
            "label": "Single source only",
            "effect": -SINGLE_SOURCE_PENALTY,
            "detail": "All evidence came from one document, so nothing independently "
                      "corroborates it.",
        })
    else:
        bonus = min(MAX_CORROBORATION_BONUS, PER_EXTRA_SOURCE_BONUS * (source_count - 1))
        score += bonus
        factors.append({
            "label": "Corroborating sources",
            "effect": round(bonus, 3),
            "detail": f"{source_count} independent sources contributed evidence.",
        })

    # 3. Contradictions - the strongest reason to doubt an answer.
    if contradictions:
        raw_penalty = sum(
            SEVERITY_PENALTY.get(c.get("severity", "medium"), SEVERITY_PENALTY["medium"])
            for c in contradictions
        )
        penalty = min(MAX_CONTRADICTION_PENALTY, raw_penalty)
        score -= penalty
        severities = ", ".join(sorted(c.get("severity", "medium") for c in contradictions))
        factors.append({
            "label": "Sources contradict each other",
            "effect": -round(penalty, 3),
            "detail": f"{len(contradictions)} conflict(s) found ({severities}).",
        })

    # 4. Citations - a grounded answer should point at its evidence.
    if answer and not _looks_like_refusal(answer) and "[source" not in answer.lower():
        score -= UNCITED_ANSWER_PENALTY
        factors.append({
            "label": "Answer is uncited",
            "effect": -UNCITED_ANSWER_PENALTY,
            "detail": "The generated answer did not cite any source, so its claims "
                      "cannot be traced back to the evidence.",
        })

    # 5. Honest refusal - cap rather than penalise, and explain the difference.
    if _looks_like_refusal(answer) and score > REFUSAL_CAP:
        score = REFUSAL_CAP
        factors.append({
            "label": "Question not answered by the evidence",
            "effect": 0.0,
            "detail": "The answer states the case file does not cover this, so "
                      "confidence reflects the gap in evidence, not the answer.",
        })

    # Round before labelling, not after: an unrounded 0.4999999999999999 from
    # floating-point arithmetic would display as 0.5 but be labelled "Low".
    score = round(max(0.0, min(1.0, score)), 3)

    return {
        "score": score,
        "percent": int(round(score * 100)),
        "label": confidence_label(score),
        "factors": factors,
    }


def confidence_label(score, has_evidence=True):
    """
    Turns the raw number into the word a user actually reads.

    "No evidence" is reserved for the case where nothing was retrieved at all.
    An answer built on real but heavily contradicted sources can bottom out at
    0.0 too, and calling that "No evidence" would be a lie - the evidence
    exists, it just disagrees with itself. That case is "Very low".
    """
    if not has_evidence:
        return "No evidence"
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.30:
        return "Low"
    return "Very low"


def build_report(question, answer, chunks, contradictions=None, evaluation=None):
    """
    Bundles the full result into one dictionary. app.py renders this; the CLI
    below prints it. Keeping assembly separate from display means both share
    exactly the same logic.
    """
    contradictions = contradictions or []
    confidence = score_confidence(chunks, contradictions, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "number": i,
                "source": chunk.get("source", "unknown"),
                "doc_type": chunk.get("doc_type", "unknown"),
                "text": chunk.get("text", ""),
                # The calibrated figure is what's shown, so the per-source
                # relevance and the confidence breakdown are on the same scale.
                "score": _calibrate_relevance(_best_score(chunk)),
                "raw_score": _best_score(chunk),
            }
            for i, chunk in enumerate(chunks, start=1)
        ],
        "contradictions": contradictions,
        "confidence": confidence,
        "evaluation": evaluation,
    }


def print_report(report):
    """Console rendering of build_report() - used by the CLI entry points."""
    print("=" * 70)
    print(f"QUESTION: {report['question']}")
    print("=" * 70)
    print(report["answer"])

    confidence = report["confidence"]
    print(f"\n--- Confidence: {confidence['label']} ({confidence['percent']}%) ---")
    for factor in confidence["factors"]:
        sign = "+" if factor["effect"] > 0 else ""
        print(f"  {sign}{factor['effect']:.2f}  {factor['label']}: {factor['detail']}")

    if report["contradictions"]:
        print("\n--- Conflicts between sources ---")
        for c in report["contradictions"]:
            print(f"  [{c['severity'].upper()}] {c['explanation']}")
            print(f"      {c['source_a']}: {c['claim_a']}")
            print(f"      {c['source_b']}: {c['claim_b']}")

    print("\n--- Sources used ---")
    for source in report["sources"]:
        print(f"  [Source {source['number']}] {source['source']} (relevance {source['score']:.0%})")
    print("=" * 70)


if __name__ == "__main__":
    from pipeline import answer_question
    from storing import get_collection

    collection = get_collection()
    result = answer_question(collection, "What certifications are found in the resume?")
    print_report(result)
