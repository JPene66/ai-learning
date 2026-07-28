"""
Objective: A reusable ingestion script that takes any source (PDF, web
article, or text file), loads it, cleans it, chunks it, and saves the
result as clean, chunk-ready records - the exact input the embedding step
(from your Week 2 study guide) needs next.

This is deliberately the connective tissue between objectives 4 and 5 -
it doesn't duplicate their logic, it just calls them in the right order.
"""

import json
import uuid
from document_loader import load_document
from text_cleaner import clean_text


def chunk_text(text, chunk_size=800, overlap=100):
    """
    Recursive-style chunking: tries to split on paragraph breaks first, and
    only falls back to a hard cut if a single paragraph is too long on its
    own. This is the same idea from your study guide's chunking concept,
    reused here as the last step before saving.
    """
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
                # a single paragraph is too big on its own - hard-split it with overlap
                start = 0
                while start < len(paragraph):
                    chunks.append(paragraph[start:start + chunk_size].strip())
                    start += chunk_size - overlap
                current = ""
            else:
                current = paragraph + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]  # drop any empty chunks


def ingest_source(source, chunk_size=800, overlap=100):
    """
    Runs one source through the full pipeline: load -> clean -> chunk.
    Returns a list of chunk records, each ready to be embedded and stored
    in a vector database in the next step.
    """
    doc = load_document(source)

    # PDFs carry page-level data, which lets clean_text remove repeated
    # headers/footers - other doc types just get whitespace/encoding cleanup.
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
            "char_count": len(chunk),
        })

    print(f"Ingested {doc['source']}: {len(chunk_records)} chunks "
          f"(from {len(cleaned_content)} cleaned characters)")

    return chunk_records


def ingest_many(sources, output_path="ingested_chunks.json", chunk_size=800, overlap=100):
    """
    Runs ingest_source() over a whole list of sources (mixing PDFs, URLs,
    and text files freely) and saves everything to one JSONL file - one
    JSON object per line, which is the standard format for feeding chunked
    data into an embedding + vector-database step.
    """
    all_chunks = []
    for source in sources:
        try:
            all_chunks.extend(ingest_source(source, chunk_size=chunk_size, overlap=overlap))
        except Exception as e:
            # One bad source (a broken URL, a corrupt PDF) shouldn't kill the
            # whole batch - log it and keep going.
            print(f"Skipped {source} due to error: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(all_chunks)} total chunks from {len(sources)} sources to {output_path}")
    return all_chunks


if __name__ == "__main__":
    sources = [
        "C:\\Users\\Jorda\\Downloads\\SJEP RS.UD.pdf",
        # "reports/quarterly_report.pdf",
        # "https://example.com/some-article",
    ]
    ingest_many(sources)