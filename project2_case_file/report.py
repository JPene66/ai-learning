"""
report.py
Generates the final "case report": a cited answer to the investigation
question, plus a confidence score based on how much the retrieved sources
actually agree with each other.
"""

from collections import Counter
from helpers import ask

REPORT_PROMPT = """You are a meticulous investigator writing a case report. Using ONLY the
evidence below, answer the investigation question. Cite the source number for every
claim you make, like [Source 1]. If sources disagree, explicitly say so rather than
picking a side. If the evidence doesn't answer the question, say so plainly.

EVIDENCE:
{evidence_block}

INVESTIGATION QUESTION: {question}
"""


def format_evidence_block(retrieved_chunks):
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        blocks.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(blocks)


def compute_confidence(retrieved_chunks, contradictions):
    """
    A simple, explainable heuristic (not a black box):
    - Start from how many DISTINCT sources were actually retrieved (more
      independent sources agreeing is stronger evidence than one source
      repeated across several chunks).
    - Subtract for every contradiction found, weighted by its severity.
    - Clamp to a 0-100 range.
    """
    distinct_sources = len(set(c["source"] for c in retrieved_chunks))
    base_score = min(100, distinct_sources * 25)  # 4+ independent sources = full base score

    severity_penalty = {"Low": 8, "Medium": 18, "High": 30}
    penalty = sum(severity_penalty.get(c["severity"], 10) for c in contradictions)

    confidence = max(0, base_score - penalty)
    return confidence


def generate_case_report(question, retrieved_chunks, contradictions):
    """
    Returns a dict with the narrative answer, a confidence score (0-100),
    and the contradictions passed through for display alongside it.
    """
    if not retrieved_chunks:
        return {
            "answer": "No evidence was found for this question. Try adding more sources or rephrasing the question.",
            "confidence": 0,
            "contradictions": [],
            "sources_used": [],
        }

    evidence_block = format_evidence_block(retrieved_chunks)
    prompt = REPORT_PROMPT.format(evidence_block=evidence_block, question=question)
    answer = ask(prompt)

    confidence = compute_confidence(retrieved_chunks, contradictions)
    sources_used = sorted(set(c["source"] for c in retrieved_chunks))

    return {
        "answer": answer,
        "confidence": confidence,
        "contradictions": contradictions,
        "sources_used": sources_used,
    }