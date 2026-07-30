"""
Tests for contradiction.py - specifically the defaults applied when parsing
the model's JSON.

Every test here uses a FAKE LLM (injected via ask_json_fn), so the suite runs
offline, costs nothing, and stays deterministic. The point isn't to test the
model - it's to test that we survive the malformed responses models really do
return: missing fields, string source numbers, invented severities, indices
pointing at sources that don't exist.
"""

from conftest import make_chunk

from contradiction import (
    detect_contradictions,
    normalize_contradictions,
    summarize_contradictions,
)

CHUNKS = [
    make_chunk("The meeting was on the 3rd.", "witness_a.pdf"),
    make_chunk("The meeting was on the 9th.", "report_b.pdf"),
]


def fake_llm(response):
    """Builds an ask_json_fn that always returns the given response."""
    return lambda prompt, **kwargs: response


def test_wellformed_response_is_parsed():
    raw = {"contradictions": [{
        "claim_a": "The meeting was on the 3rd.",
        "source_a": 1,
        "claim_b": "The meeting was on the 9th.",
        "source_b": 2,
        "severity": "high",
        "explanation": "Two different dates for the same meeting.",
    }]}

    found = normalize_contradictions(raw, CHUNKS)

    assert len(found) == 1
    assert found[0]["source_a"] == "witness_a.pdf"
    assert found[0]["source_b"] == "report_b.pdf"
    assert found[0]["severity"] == "high"


def test_missing_severity_defaults_to_medium():
    raw = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 2,
    }]}
    assert normalize_contradictions(raw, CHUNKS)[0]["severity"] == "medium"


def test_invented_severity_is_mapped_or_defaulted():
    aliased = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 2,
        "severity": "CRITICAL",
    }]}
    unknown = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 2,
        "severity": "spicy",
    }]}

    assert normalize_contradictions(aliased, CHUNKS)[0]["severity"] == "high"
    assert normalize_contradictions(unknown, CHUNKS)[0]["severity"] == "medium"


def test_missing_explanation_gets_a_readable_default():
    raw = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 2,
    }]}
    assert normalize_contradictions(raw, CHUNKS)[0]["explanation"]


def test_source_numbers_as_strings_still_resolve():
    raw = {"contradictions": [{
        "claim_a": "A", "source_a": "1", "claim_b": "B", "source_b": "2",
    }]}
    found = normalize_contradictions(raw, CHUNKS)
    assert found[0]["source_a"] == "witness_a.pdf"


def test_out_of_range_source_number_does_not_crash():
    raw = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 99,
    }]}
    found = normalize_contradictions(raw, CHUNKS)
    assert len(found) == 1
    assert "unknown source" in found[0]["source_b"]


def test_one_sided_contradiction_is_dropped():
    raw = {"contradictions": [
        {"claim_a": "A", "source_a": 1, "claim_b": "", "source_b": 2},
        {"claim_a": "", "source_a": 1, "claim_b": "B", "source_b": 2},
    ]}
    assert normalize_contradictions(raw, CHUNKS) == []


def test_same_source_conflict_is_dropped():
    """Two passages from one document isn't a cross-source contradiction."""
    raw = {"contradictions": [{
        "claim_a": "A", "source_a": 1, "claim_b": "B", "source_b": 1,
    }]}
    assert normalize_contradictions(raw, CHUNKS) == []


def test_garbage_shapes_return_empty_list():
    assert normalize_contradictions(None, CHUNKS) == []
    assert normalize_contradictions({}, CHUNKS) == []
    assert normalize_contradictions({"contradictions": "nope"}, CHUNKS) == []
    assert normalize_contradictions({"contradictions": ["a string"]}, CHUNKS) == []


def test_single_source_skips_the_llm_entirely():
    """A contradiction needs two sources, so there's nothing to spend a call on."""
    def explode(prompt, **kwargs):
        raise AssertionError("the LLM should not have been called")

    one_source = [make_chunk("A", "only.pdf"), make_chunk("B", "only.pdf")]
    assert detect_contradictions("q", one_source, ask_json_fn=explode) == []
    assert detect_contradictions("q", [], ask_json_fn=explode) == []


def test_llm_failure_degrades_to_no_contradictions():
    def broken(prompt, **kwargs):
        raise RuntimeError("API down")

    assert detect_contradictions("q", CHUNKS, ask_json_fn=broken) == []


def test_detect_contradictions_end_to_end_with_fake_llm():
    response = {"contradictions": [{
        "claim_a": "The meeting was on the 3rd.", "source_a": 1,
        "claim_b": "The meeting was on the 9th.", "source_b": 2,
        "severity": "high", "explanation": "Different dates.",
    }]}

    found = detect_contradictions("When was the meeting?", CHUNKS,
                                  ask_json_fn=fake_llm(response))

    assert len(found) == 1
    assert found[0]["severity"] == "high"


def test_summary_line():
    assert "No conflicts" in summarize_contradictions([])

    one = normalize_contradictions(
        {"contradictions": [{"claim_a": "A", "source_a": 1,
                             "claim_b": "B", "source_b": 2, "severity": "high"}]},
        CHUNKS,
    )
    summary = summarize_contradictions(one)
    assert "1 conflict found" in summary
    assert "1 high severity" in summary
