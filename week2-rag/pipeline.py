"""
pipeline.py
The connective tissue: wires every step of both lanes into two functions the
UI (or a script) can call.

    INGESTION LANE
    load_document -> clean_text -> chunk_text -> embed_many -> store_chunks

    QUERY LANE
    rewrite_followup -> hybrid_search -> rerank -> build_prompt -> call_llm
                     -> detect_contradictions -> score_confidence -> build_report

Nothing new is implemented here. Every step already exists in its own file
with its own explanation; this file only calls them in the right order, which
means app.py stays a thin user interface instead of hiding pipeline logic
inside Streamlit callbacks.
"""

from helpers import embed_many
from ingest import ingest_source
from storing import store_chunks, delete_source
from search import hybrid_search
from rerank import rerank
from context_assembly import build_prompt
from llm_call import call_llm
from contradiction import detect_contradictions
from rag_eval import evaluate_answer
from multiturn import rewrite_followup
from report import build_report


def ingest_and_store(collection, source, chunk_size=800, overlap=100, source_name=None):
    """
    Runs one source (PDF path, .txt/.md path, or http(s) URL) all the way into
    the vector store, and reports what happened at each stage.

    Embeddings are generated EXPLICITLY here via helpers.embed_many() rather
    than letting Chroma embed documents automatically. That's a deliberate
    choice: the query side (query_embed.py) uses the same helpers.embed(), so
    both sides of the search provably use the same model. Handing embedding
    off to Chroma's default would silently use a different model on the
    document side than the rest of this project uses on the query side.
    """
    chunk_records = ingest_source(
        source, chunk_size=chunk_size, overlap=overlap, source_name=source_name
    )

    if not chunk_records:
        raise ValueError(
            f"No readable text found in {source_name or source}. If it's a PDF, it "
            "may be a scanned image with no text layer - those need OCR first."
        )

    vectors = embed_many([c["text"] for c in chunk_records])
    for record, vector in zip(chunk_records, vectors):
        record["embedding"] = vector

    label = chunk_records[0]["source"]

    # Clear any previous version of this document first. Chunk IDs are stable,
    # so re-ingesting overwrites matching chunks - but if the document got
    # SHORTER, the leftover tail chunks from the old version would still be
    # sitting in the collection, quietly citable as current evidence.
    delete_source(collection, label)

    stored = store_chunks(collection, chunk_records)

    return {
        "source": label,
        "doc_type": chunk_records[0]["doc_type"],
        "chunks_stored": stored,
        "characters": sum(c["char_count"] for c in chunk_records),
    }


def answer_question(
    collection,
    question,
    history=None,
    n_candidates=12,
    keep_top=4,
    detect_conflicts=True,
    evaluate=False,
):
    """
    The full query lane. Returns the report dict from report.build_report(),
    with two extra keys the UI uses to show its work:

        retrieval_query - the question actually used for retrieval, which
                          differs from `question` when a follow-up was rewritten
        candidates      - everything hybrid search found, before re-ranking
    """
    # 1. Multi-turn: resolve "and what about that one?" into a real question.
    retrieval_query = rewrite_followup(question, history)

    # 2. Hybrid retrieval: vector + keyword, blended.
    candidates = hybrid_search(collection, retrieval_query, n_results=n_candidates)

    # 3. Re-rank the shortlist by true relevance.
    top_chunks = rerank(retrieval_query, candidates, keep_top=keep_top) if candidates else []

    # 4. Assemble labeled context and generate the cited answer.
    prompt = build_prompt(question, top_chunks, conversation_history=history)
    answer = call_llm(prompt)

    # 5. Audit the evidence for cross-source disagreement.
    contradictions = (
        detect_contradictions(retrieval_query, top_chunks) if detect_conflicts else []
    )

    # 6. Optional self-grading.
    evaluation = evaluate_answer(question, answer, top_chunks) if evaluate else None

    # 7. Score confidence and bundle everything for display.
    report = build_report(question, answer, top_chunks, contradictions, evaluation)
    report["retrieval_query"] = retrieval_query
    report["candidates"] = candidates
    return report


if __name__ == "__main__":
    from storing import get_collection
    from report import print_report

    collection = get_collection()
    print_report(answer_question(collection, "What is this case file about?"))
