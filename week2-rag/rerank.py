"""
(QUERY lane): take the shortlist of candidates
from search.py and re-order them by TRUE relevance to the question, using
an LLM as a more careful (but slower) judge than similarity search alone.

Input:  a list of candidate chunks (from search.py) + the original question
Output: the same chunks, re-ordered - and often trimmed down to a smaller
        final set, since re-ranking is comparatively expensive and should
        only run on a short list, never the whole collection.
"""

from helpers import ask_json
from search import hybrid_search
from storing import get_collection

# How much of each candidate the judge sees. Full chunks would blow up the
# prompt when re-ranking 18 candidates; the opening of a chunk is nearly always
# enough to judge relevance.
PREVIEW_CHARS = 500


def rerank(question, candidates, keep_top=4, ask_json_fn=None):
    """
    Asks the LLM to rank the candidates by genuine relevance to the
    question - not just similarity score - and returns them re-ordered,
    trimmed to keep_top. This catches cases where a technically-close
    vector match isn't actually the most USEFUL answer to what was asked.

    ask_json_fn exists so tests can pass a fake LLM instead of calling the
    real API - the default is the real one.
    """
    if len(candidates) <= 1:
        return candidates  # nothing to meaningfully re-rank

    ask_json_fn = ask_json_fn or ask_json

    numbered = "\n\n".join(
        f"{i}: {c['text'][:PREVIEW_CHARS]}" for i, c in enumerate(candidates)
    )
    prompt = f"""Question: {question}

Candidates:
{numbered}

Rank these candidates from MOST to LEAST relevant to answering the question.
Include every candidate number exactly once.
Respond ONLY as JSON: {{"ranking": [candidate numbers, most relevant first]}}"""

    try:
        result = ask_json_fn(prompt)
    except Exception as e:
        # Re-ranking is a refinement, not a requirement. If it fails, fall back
        # to the search ordering rather than failing the user's whole question.
        print(f"Re-ranking failed ({e}) - falling back to search order.")
        return candidates[:keep_top]

    ranking = result.get("ranking", []) if isinstance(result, dict) else []

    # Defensive: models sometimes return "2" instead of 2, repeat an index, or
    # return one outside the list. Coerce, de-duplicate, and drop invalid ones.
    seen = set()
    valid_ranking = []
    for item in ranking:
        try:
            i = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(candidates) and i not in seen:
            seen.add(i)
            valid_ranking.append(i)

    # Any candidate the model forgot to mention keeps its original relative
    # position at the end, so nothing is silently lost.
    valid_ranking += [i for i in range(len(candidates)) if i not in seen]

    reranked = [candidates[i] for i in valid_ranking]
    return reranked[:keep_top]


if __name__ == "__main__":
    collection = get_collection()
    question = "What certifications are there in the resume"

    candidates = hybrid_search(collection, question, n_results=6)
    print(f"Before re-ranking ({len(candidates)} candidates):")
    for c in candidates:
        print(f"  [{c['blended_score']}] {c['source']}")

    reranked = rerank(question, candidates, keep_top=4)
    print(f"\nAfter re-ranking (top {len(reranked)}):")
    for c in reranked:
        print(f"  {c['source']} - {c['text'][:70]}")
