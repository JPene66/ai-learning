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

from helpers import embed
from storing import get_collection


def vector_search(collection, query, n_results=6):
    """
    Pure similarity search: embed the query, ask Chroma for the closest
    vectors. Fast and usually good, but can miss exact terms (names, dates,
    specific numbers) if the wording is unusual.
    """
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    candidates = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "doc_type": meta.get("doc_type", "unknown"),
            "vector_similarity": round(1 / (1 + distance), 3),  # convert distance -> 0-1 similarity
            "keyword_score": None,
            "blended_score": None,
        })
    return candidates


def keyword_score(text, query):
    """Fraction of the query's meaningful words that literally appear in this chunk."""
    query_words = {w for w in query.lower().split() if len(w) > 2}
    if not query_words:
        return 0.0
    text_words = set(text.lower().split())
    return len(query_words & text_words) / len(query_words)


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
                "source": meta.get("source", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
                "vector_similarity": None,
                "keyword_score": round(score, 3),
                "blended_score": None,
            })
    return sorted(scored, key=lambda c: c["keyword_score"], reverse=True)[:n_results]


def hybrid_search(collection, query, n_results=6, vector_weight=0.7):
    """
    The one you'll actually use in the project: retrieves candidates by
    vector similarity, then re-scores each one by blending in a keyword
    score. vector_weight=1.0 would be pure vector search, 0.0 pure keyword.
    """
    candidates = vector_search(collection, query, n_results=n_results)
    for c in candidates:
        kw = keyword_score(c["text"], query)
        c["keyword_score"] = round(kw, 3)
        c["blended_score"] = round((vector_weight * c["vector_similarity"]) + ((1 - vector_weight) * kw), 3)
    return sorted(candidates, key=lambda c: c["blended_score"], reverse=True)


if __name__ == "__main__":
    collection = get_collection()
    query = "What certications are there in the resume"

    print("=== VECTOR SEARCH ===")
    for r in vector_search(collection, query, n_results=3):
        print(f"[{r['vector_similarity']}] {r['source']} - {r['text'][:70]}")

    print("\n=== KEYWORD SEARCH ===")
    for r in keyword_search(collection, query, n_results=3):
        print(f"[{r['keyword_score']}] {r['source']} - {r['text'][:70]}")

    print("\n=== HYBRID SEARCH ===")
    for r in hybrid_search(collection, query, n_results=3):
        print(f"[{r['blended_score']}] {r['source']} - {r['text'][:70]}")