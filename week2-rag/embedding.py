"""
the RAG pipeline (Ingestion lane): turn each chunk's text into a
vector using the embedding model.

Input:  ingested_chunks.json  (produced by ingest.py - one JSON chunk record per line)
Output: embedded_chunks.json  (same records, each with an added "embedding" field)

This is deliberately its own file/step: embedding is a distinct, separate
concept from chunking, and it's also the SAME function will be reuse
later for embedding the user's query, one function, two uses.
"""

import json
from helpers import embed


def load_chunk_records(input_path="ingested_chunks.json"):
    """Reads the JSONL file produced by ingest.py back into a list of dicts."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_chunks(chunk_records):
    """
    Adds an 'embedding' field to every chunk record. This is the single most
    expensive step in the ingestion pipeline (one API call per chunk), which
    is exactly why it's kept separate from storing - if storing fails, you
    don't want to have to re-embed everything.
    """
    for record in chunk_records:
        record["embedding"] = embed(record["text"])
    return chunk_records


def save_embedded_chunks(chunk_records, output_path="embedded_chunks.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(chunk_records)} embedded chunks to {output_path}")


if __name__ == "__main__":
    chunks = load_chunk_records("ingested_chunks.json")
    print(f"Loaded {len(chunks)} chunks to embed")

    embedded = embed_chunks(chunks)
    print(f"Embedded {len(embedded)} chunks (vector length: {len(embedded[0]['embedding'])})")

    save_embedded_chunks(embedded)