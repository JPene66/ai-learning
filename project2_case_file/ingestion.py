"""
ingestion.py
Takes any source (PDF, web article, or text file), loads it, cleans it, and
chunks it - the same logic from Day 1's ingest.py, adapted here to hand
chunks directly to the vector store instead of writing them to a file.
"""

import uuid
from document_loader import load_document
from text_cleaner import clean_text


def chunk_text(text, chunk_size=600, overlap=80):
    """Paragraph-aware chunking with a hard-split fallback for oversized paragraphs."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) <= chunk_size:
            current += paragraph + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(paragraph) > chunk_size:
                start = 0
                while start < len(paragraph):
                    chunks.append(paragraph[start:start + chunk_size].strip())
                    start += chunk_size - overlap
                current = ""
            else:
                current = paragraph + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def ingest_source(source, chunk_size=600, overlap=80):
    """Loads, cleans, and chunks one source. Returns a list of chunk records."""
    doc = load_document(source)

    pages = doc["metadata"].get("pages") if doc["doc_type"] == "pdf" else None
    cleaned_content = clean_text(doc["content"], pages=pages)

    raw_chunks = chunk_text(cleaned_content, chunk_size=chunk_size, overlap=overlap)

    chunk_records = []
    for i, chunk in enumerate(raw_chunks):
        chunk_records.append({
            "chunk_id": str(uuid.uuid4()),
            "source": doc["source"],
            "doc_type": doc["doc_type"],
            "chunk_index": i,
            "text": chunk,
        })

    return chunk_records