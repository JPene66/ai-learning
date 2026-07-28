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


def rerank(question, candidates, keep_top=4):
    """
    Asks the LLM to rank the candidates by genuine relevance to the
    question - not just similarity score - and returns them re-ordered,
    trimmed to keep_top. This catches cases where a technically-close
    vector match isn't actually the most USEFUL answer to what was asked.
    """
    if len(candidates) <= 1:
        return candidates  # nothing to meaningfully re-rank

    numbered = "\n".join(f"{i}: {c['text']}" for i, c in enumerate(candidates))
    prompt = f"""Question: {question}

Candidates:
{numbered}

Rank these candidates from MOST to LEAST relevant to answering the question.
Respond ONLY as JSON: {{"ranking": [candidate numbers, most relevant first]}}"""

    result = ask_json(prompt)
    ranking = result.get("ranking", list(range(len(candidates))))  # fall back to original order if parsing fails

    # Defensive: only keep ranking indices that are actually valid, in case
    # the model returns a number outside the candidate list.
    valid_ranking = [i for i in ranking if isinstance(i, int) and 0 <= i < len(candidates)]
    reranked = [candidates[i] for i in valid_ranking]

    return reranked[:keep_top]


if __name__ == "__main__":
    collection = get_collection()
    question = "What certications are there in the resume"

    candidates = hybrid_search(collection, question, n_results=6)
    print(f"Before re-ranking ({len(candidates)} candidates):")
    for c in candidates:
        print(f"  [{c['blended_score']}] {c['source']}")

    reranked = rerank(question, candidates, keep_top=4)
    print(f"\nAfter re-ranking (top {len(reranked)}):")
    for c in reranked:
        print(f"  {c['source']} - {c['text'][:70]}")