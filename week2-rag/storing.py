"""
Storing phase of the RAG pipeline (Ingestion lane): store the embedded chunks in a
vector database (Chroma) so they can actually be searched later.

Input:  embedded_chunks.json (produced by embedding.py)
Output: a persistent Chroma collection on disk at ./chroma_db

This is the last step of the INGESTION pipeline. Everything from here
onward (query_embed.py, search.py, rerank.py...) belongs to the QUERY
pipeline, and reads FROM this stored collection instead of from files.
"""

import json
import chromadb

DEFAULT_COLLECTION = "case_file_ai"

# Chroma's default distance metric is L2 (squared euclidean). We explicitly ask
# for cosine instead, because every "similarity score" shown in this project is
# computed as (1 - distance), and that formula is only meaningful for cosine
# distance, which is bounded in [0, 2]. With L2 the number would be unbounded
# and the displayed score would be nonsense. Setting it here, once, is what
# makes search.py's conversion honest.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def load_embedded_chunks(input_path="embedded_chunks.json"):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_collection(persist_path="./chroma_db", name=DEFAULT_COLLECTION):
    """
    A PersistentClient writes to disk - this is what makes the collection
    still be there tomorrow, unlike chromadb.Client() which is in-memory
    and disappears the moment the script ends.
    """
    client_db = chromadb.PersistentClient(path=persist_path)
    collection = client_db.get_or_create_collection(name, metadata=COLLECTION_METADATA)

    # get_or_create returns an EXISTING collection untouched, so a collection
    # created before this metadata was added would silently still be on L2.
    # Warn rather than fail, so old data stays readable - but the user knows
    # the similarity scores from it aren't directly comparable.
    space = (collection.metadata or {}).get("hnsw:space")
    if space != "cosine":
        print(
            f"Warning: collection '{name}' uses '{space or 'l2'}' distance, not cosine. "
            "Similarity scores will be unreliable - delete ./chroma_db and re-ingest "
            "to rebuild it with cosine distance."
        )

    return collection


def store_chunks(collection, chunk_records, batch_size=100):
    """
    Chroma wants 4 parallel lists (documents, embeddings, metadatas, ids) -
    not a list of dicts - so this function's whole job is reshaping our
    chunk records into the shape Chroma expects.

    Two deliberate choices:
    - upsert() instead of add(), so re-ingesting the same document updates the
      existing chunks instead of raising a duplicate-ID error.
    - batching, because Chroma rejects very large single writes and a long PDF
      can easily produce more chunks than one request allows.
    """
    if not chunk_records:
        return 0

    for start in range(0, len(chunk_records), batch_size):
        batch = chunk_records[start:start + batch_size]
        collection.upsert(
            documents=[c["text"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "doc_type": c["doc_type"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
            ids=[c["chunk_id"] for c in batch],
        )

    return len(chunk_records)


def list_sources(collection):
    """
    Returns {source_name: chunk_count} for everything currently in the
    collection. This is what lets the app show a "case file" panel listing
    the evidence it's actually working from, instead of a mystery blob of
    chunks.
    """
    stored = collection.get(include=["metadatas"])
    counts = {}
    for meta in stored.get("metadatas") or []:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def delete_source(collection, source):
    """Removes every chunk belonging to one source - lets a user drop a single
    piece of evidence from the case without wiping the whole database."""
    collection.delete(where={"source": source})


if __name__ == "__main__":
    chunks = load_embedded_chunks("embedded_chunks.json")
    print(f"Loaded {len(chunks)} embedded chunks to store")

    collection = get_collection()
    store_chunks(collection, chunks)

    print(f"Collection now contains {collection.count()} total chunks")
    print("Stored at: ./chroma_db (persists across script runs)")
