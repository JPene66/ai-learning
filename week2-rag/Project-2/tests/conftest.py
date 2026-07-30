"""
Shared test setup: put the pipeline modules (one folder above Project-2) on
the import path, and provide the chunk fixtures the tests reuse.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PIPELINE_ROOT)


# Scores here are RAW retrieval scores, deliberately in the range real hybrid
# search actually produces (~0.2 for unrelated text, ~0.6 for an excellent
# match) rather than a tidy 0.9 - otherwise the tests would only ever exercise
# the saturated end of report.py's calibration curve.
STRONG, GOOD, FAIR = 0.60, 0.52, 0.44


def make_chunk(text="some evidence", source="witness_a.pdf", score=GOOD):
    return {
        "text": text,
        "source": source,
        "doc_type": "pdf",
        "chunk_index": 0,
        "vector_similarity": score,
        "keyword_score": 0.5,
        "blended_score": score,
    }


@pytest.fixture
def two_source_chunks():
    """Strong evidence from two independent documents.
    Calibrates to 1.00 / 0.80 / 0.60 -> evidence strength 0.80."""
    return [
        make_chunk("The meeting was on the 3rd.", "witness_a.pdf", STRONG),
        make_chunk("Attendees confirmed the 3rd.", "report_b.pdf", GOOD),
        make_chunk("The 3rd is listed in the log.", "report_b.pdf", FAIR),
    ]


@pytest.fixture
def single_source_chunks():
    """The same evidence quality, but all from one document."""
    return [
        make_chunk("The meeting was on the 3rd.", "witness_a.pdf", STRONG),
        make_chunk("It started at nine.", "witness_a.pdf", GOOD),
        make_chunk("Six people attended.", "witness_a.pdf", FAIR),
    ]


def contradiction(severity="high"):
    return {
        "claim_a": "The meeting was on the 3rd.",
        "source_a": "witness_a.pdf",
        "claim_b": "The meeting was on the 9th.",
        "source_b": "report_b.pdf",
        "severity": severity,
        "explanation": "The two sources give different dates.",
    }
