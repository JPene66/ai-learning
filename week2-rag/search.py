"""
search.py
STEP 5 of the RAG pipeline (QUERY lane): search the vector database for
chunks relevant to the embedded query.

This file deliberately shows THREE search methods side by side, so
students can see the actual difference in results between them, not just
read about it:

    1. vector_search()   - pure similarity search (embeddings only)
    2. keyword_search()  - pure keyword overlap (no embeddings at all)
    3. hybrid_search()   - blends both together

Input:  a query string + the stored collection (from storing.py)
Output: a ranked list of candidate chunks - the shortlist that rerank.py
        (step 6) will refine further.
"""

import re

from helpers import embed
from storing import get_collection

# Words so common they'd match almost any chunk and inflate keyword scores.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "what", "when", "where", "which", "who", "how", "why", "does", "did", "has",
    "have", "had", "into", "about", "there", "their", "they", "you", "your",
    "its", "it's", "but", "not", "all", "any", "can", "will", "would", "should",
}


def _similarity_from_distance(distance):
    """
    Converts Chroma's cosine DISTANCE into a 0-1 SIMILARITY score.

    Cosine distance = 1 - cosine_similarity, so it runs from 0 (identical) to 2
    (opposite). Subtracting from 1 gives back the cosine similarity, which we
    then clamp to [0, 1] because negative similarity ("actively unrelated") is
    not a useful thing to show a user - it's just zero relevance.

    This is only correct because storing.py explicitly creates the collection
    with cosine distance. Change that, and this function has to change too.
    """
    similarity = 1 - distance
    return round(max(0.0, min(1.0, similarity)), 3)


def vector_search(collection, query, n_results=6):
    """
    Pure similarity search: embed the query, ask Chroma for the closest
    vectors. Fast and usually good, but can miss exact terms (names, dates,
    specific numbers) if the wording is unusual.
    """
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({
            "text": doc,
            "source": (meta or {}).get("source", "unknown"),
            "doc_type": (meta or {}).get("doc_type", "unknown"),
            "chunk_index": (meta or {}).get("chunk_index"),
            "vector_similarity": _similarity_from_distance(distance),
            "keyword_score": None,
            "blended_score": None,
        })
    return candidates


def keyword_score(text, query):
    """
    Fraction of the query's meaningful words that literally appear in this chunk.

    Words are split on non-letters/digits so "GDPR," and "GDPR" count as the
    same word, and stopwords are dropped so a chunk doesn't score well just for
    containing "the" and "what".
    """
    def tokenize(s):
        return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOPWORDS}

    query_words = tokenize(query)
    if not query_words:
        return 0.0
    return len(query_words & tokenize(text)) / len(query_words)


def keyword_search(collection, query, n_results=6):
    """
    Pure keyword search: no embeddings involved at all. Scans every stored
    chunk and scores it purely by word overlap. Slower to scale, but never
    misses an exact term match the way vector search sometimes can.
    """
    all_data = collection.get(include=["documents", "metadatas"])
    scored = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        score = keyword_score(doc, query)
        if score > 0:
            scored.append({
                "text": doc,
                "source": (meta or {}).get("source", "unknown"),
                "doc_type": (meta or {}).get("doc_type", "unknown"),
                "chunk_index": (meta or {}).get("chunk_index"),
                "vector_similarity": None,
                "keyword_score": round(score, 3),
                "blended_score": None,
            })
    return sorted(scored, key=lambda c: c["keyword_score"], reverse=True)[:n_results]


def hybrid_search(collection, query, n_results=6, vector_weight=0.7, pool_multiplier=3):
    """
    The one you'll actually use in the project.

    A true blend, not just re-scored vector results: we pull a WIDER pool from
    vector search, pull the top keyword hits separately, and merge the two.
    That matters because a chunk containing the exact term you asked about but
    phrased differently can be missed entirely by vector search - if we only
    re-scored vector results, keyword scoring could never rescue it.

    Every candidate then gets a blended score:
        (vector_weight * vector_similarity) + ((1 - vector_weight) * keyword_score)

    vector_weight=1.0 would be pure vector search, 0.0 pure keyword.
    """
    pool_size = n_results * pool_multiplier

    merged = {}
    for candidate in vector_search(collection, query, n_results=pool_size):
        merged[candidate["text"]] = candidate

    for candidate in keyword_search(collection, query, n_results=pool_size):
        # A chunk found by both methods keeps its vector similarity; one found
        # only by keyword search has no vector score, so it counts as 0.
        if candidate["text"] not in merged:
            merged[candidate["text"]] = candidate

    candidates = list(merged.values())
    for c in candidates:
        vector_part = c["vector_similarity"] or 0.0
        kw = keyword_score(c["text"], query)
        c["vector_similarity"] = vector_part
        c["keyword_score"] = round(kw, 3)
        c["blended_score"] = round((vector_weight * vector_part) + ((1 - vector_weight) * kw), 3)

    return sorted(candidates, key=lambda c: c["blended_score"], reverse=True)[:n_results]


if __name__ == "__main__":
    collection = get_collection()
    query = "What certifications are there in the resume"

    print("=== VECTOR SEARCH ===")
    for r in vector_search(collection, query, n_results=3):
        print(f"[{r['vector_similarity']}] {r['source']} - {r['text'][:70]}")

    print("\n=== KEYWORD SEARCH ===")
    for r in keyword_search(collection, query, n_results=3):
        print(f"[{r['keyword_score']}] {r['source']} - {r['text'][:70]}")

    print("\n=== HYBRID SEARCH ===")
    for r in hybrid_search(collection, query, n_results=3):
        print(f"[{r['blended_score']}] {r['source']} - {r['text'][:70]}")
