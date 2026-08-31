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


def load_embedded_chunks(input_path="embedded_chunks.json"):
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_collection(persist_path="./chroma_db", name="week2_workshop"):
    """
    A PersistentClient writes to disk - this is what makes the collection
    still be there tomorrow, unlike chromadb.Client() which is in-memory
    and disappears the moment the script ends.
    """
    client_db = chromadb.PersistentClient(path=persist_path)
    return client_db.get_or_create_collection(name)


def store_chunks(collection, chunk_records):
    """
    Chroma wants 4 parallel lists (documents, embeddings, metadatas, ids) -
    not a list of dicts - so this function's whole job is reshaping our
    chunk records into the shape Chroma expects.
    """
    collection.add(
        documents=[c["text"] for c in chunk_records],
        embeddings=[c["embedding"] for c in chunk_records],
        metadatas=[
            {"source": c["source"], "doc_type": c["doc_type"], "chunk_index": c["chunk_index"]}
            for c in chunk_records
        ],
        ids=[c["chunk_id"] for c in chunk_records],
    )


if __name__ == "__main__":
    chunks = load_embedded_chunks("embedded_chunks.json")
    print(f"Loaded {len(chunks)} embedded chunks to store")

    collection = get_collection()
    store_chunks(collection, chunks)

    print(f"Collection now contains {collection.count()} total chunks")
    print("Stored at: ./chroma_db (persists across script runs)")