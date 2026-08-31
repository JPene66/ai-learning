"""
vectorstore.py
Wraps ChromaDB and adds hybrid search - blending vector similarity with a
plain keyword-overlap score, since exact terms (names, dates, specific
numbers) are sometimes under-weighted by embeddings alone.
"""

import chromadb
from helpers import embed
from ingestion import ingest_source


def get_collection(persist_path="./chroma_db", name="case_evidence"):
    client_db = chromadb.PersistentClient(path=persist_path)
    return client_db.get_or_create_collection(name)


def add_source_to_case(collection, source):
    """Ingests one source (PDF/URL/text file) and stores its chunks."""
    chunks = ingest_source(source)
    if not chunks:
        return 0

    collection.add(
        documents=[c["text"] for c in chunks],
        embeddings=[embed(c["text"]) for c in chunks],
        metadatas=[{"source": c["source"], "doc_type": c["doc_type"]} for c in chunks],
        ids=[c["chunk_id"] for c in chunks],
    )
    return len(chunks)


def keyword_score(text, query):
    """Fraction of the query's words that literally appear in this chunk."""
    query_words = set(w for w in query.lower().split() if len(w) > 2)
    if not query_words:
        return 0.0
    text_words = set(text.lower().split())
    return len(query_words & text_words) / len(query_words)


def hybrid_search(collection, query, n_results=6, vector_weight=0.7):
    """
    Retrieves candidates by vector similarity, then re-scores them by
    blending in a keyword-overlap score. Returns a list of dicts, each with
    the chunk text, its source metadata, and the blended score - sorted
    best-first.

    vector_weight controls the blend: 1.0 = pure vector search,
    0.0 = pure keyword search.
    """
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    scored = []
    for doc, meta, distance in zip(docs, metas, distances):
        # Chroma returns a distance (lower = more similar); convert to a
        # 0-1 similarity score so it can be blended with the keyword score.
        vector_similarity = 1 / (1 + distance)
        kw_score = keyword_score(doc, query)
        blended = (vector_weight * vector_similarity) + ((1 - vector_weight) * kw_score)
        scored.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "doc_type": meta.get("doc_type", "unknown"),
            "vector_similarity": round(vector_similarity, 3),
            "keyword_score": round(kw_score, 3),
            "blended_score": round(blended, 3),
        })

    return sorted(scored, key=lambda r: r["blended_score"], reverse=True)