"""
Tests for the confidence-scoring heuristic in report.py.

This is the most important thing in the project to test, because it's the one
number a user is asked to trust and it's computed by us, not by the model. A
silent sign error here would make the app confidently wrong.
"""

from conftest import STRONG, contradiction, make_chunk

from report import (
    MAX_CONTRADICTION_PENALTY,
    REFUSAL_CAP,
    RELEVANCE_CEILING,
    RELEVANCE_FLOOR,
    SINGLE_SOURCE_PENALTY,
    UNCITED_ANSWER_PENALTY,
    _calibrate_relevance,
    build_report,
    confidence_label,
    score_confidence,
)

CITED_ANSWER = "The meeting was on the 3rd [Source 1]."

# Evidence strength of the two_source_chunks / single_source_chunks fixtures.
EVIDENCE_STRENGTH = 0.80


def test_no_chunks_scores_zero():
    result = score_confidence([], [], "anything")
    assert result["score"] == 0.0
    assert result["percent"] == 0
    assert result["label"] == "No evidence"
    assert result["factors"], "an explanation is required even at zero"


def test_raw_retrieval_scores_are_calibrated():
    """Raw cosine scores under-report relevance; the floor/ceiling stretch fixes it."""
    assert _calibrate_relevance(RELEVANCE_FLOOR) == 0.0
    assert _calibrate_relevance(RELEVANCE_CEILING) == 1.0
    assert _calibrate_relevance(0.10) == 0.0        # unrelated text, clamped
    assert _calibrate_relevance(0.95) == 1.0        # better than the ceiling, clamped
    assert _calibrate_relevance(0.40) == 0.5        # midpoint


def test_strong_multi_source_evidence_scores_high(two_source_chunks):
    result = score_confidence(two_source_chunks, [], CITED_ANSWER)
    # calibrated top 3 (1.00, 0.80, 0.60) = 0.80, +0.05 for a second source
    assert result["score"] == round(EVIDENCE_STRENGTH + 0.05, 3)
    assert result["label"] == "High"


def test_single_source_is_penalised(single_source_chunks, two_source_chunks):
    single = score_confidence(single_source_chunks, [], CITED_ANSWER)
    multi = score_confidence(two_source_chunks, [], CITED_ANSWER)

    assert single["score"] < multi["score"]
    assert single["score"] == round(EVIDENCE_STRENGTH - SINGLE_SOURCE_PENALTY, 3)
    assert any("Single source" in f["label"] for f in single["factors"])


def test_corroboration_bonus_is_capped():
    chunks = [make_chunk(f"evidence {i}", f"source_{i}.pdf", STRONG) for i in range(10)]
    result = score_confidence(chunks, [], CITED_ANSWER)
    bonus = next(f["effect"] for f in result["factors"] if f["label"] == "Corroborating sources")
    assert bonus == 0.15


def test_contradictions_reduce_confidence(two_source_chunks):
    clean = score_confidence(two_source_chunks, [], CITED_ANSWER)
    conflicted = score_confidence(two_source_chunks, [contradiction("high")], CITED_ANSWER)

    assert conflicted["score"] < clean["score"]
    assert conflicted["score"] == 0.50
    assert conflicted["label"] == "Moderate"


def test_higher_severity_costs_more(two_source_chunks):
    high = score_confidence(two_source_chunks, [contradiction("high")], CITED_ANSWER)
    medium = score_confidence(two_source_chunks, [contradiction("medium")], CITED_ANSWER)
    low = score_confidence(two_source_chunks, [contradiction("low")], CITED_ANSWER)

    assert high["score"] < medium["score"] < low["score"]


def test_unknown_severity_falls_back_to_medium(two_source_chunks):
    weird = score_confidence(two_source_chunks, [contradiction("catastrophic")], CITED_ANSWER)
    medium = score_confidence(two_source_chunks, [contradiction("medium")], CITED_ANSWER)
    assert weird["score"] == medium["score"]


def test_contradiction_penalty_is_capped(two_source_chunks):
    many = [contradiction("high") for _ in range(6)]
    result = score_confidence(two_source_chunks, many, CITED_ANSWER)

    penalty = next(f["effect"] for f in result["factors"]
                   if f["label"] == "Sources contradict each other")
    assert penalty == -MAX_CONTRADICTION_PENALTY
    assert result["score"] == round(EVIDENCE_STRENGTH + 0.05 - MAX_CONTRADICTION_PENALTY, 3)


def test_uncited_answer_is_penalised(two_source_chunks):
    uncited = score_confidence(two_source_chunks, [], "The meeting was on the 3rd.")
    cited = score_confidence(two_source_chunks, [], CITED_ANSWER)

    assert uncited["score"] == round(cited["score"] - UNCITED_ANSWER_PENALTY, 3)
    assert any("uncited" in f["label"].lower() for f in uncited["factors"])


def test_honest_refusal_is_capped_not_penalised(two_source_chunks):
    refusal = "The case file contains no evidence about the meeting date."
    result = score_confidence(two_source_chunks, [], refusal)

    assert result["score"] == REFUSAL_CAP
    # A refusal has no citations, but shouldn't also be dinged for that.
    assert not any("uncited" in f["label"].lower() for f in result["factors"])


def test_score_never_leaves_zero_to_one_range():
    weak = [make_chunk("thin evidence", "only.pdf", 0.05)]
    result = score_confidence(weak, [contradiction("high")] * 5, "Uncited claim.")
    assert 0.0 <= result["score"] <= 1.0
    assert result["percent"] == int(round(result["score"] * 100))


def test_confidence_label_bands():
    assert confidence_label(0.95) == "High"
    assert confidence_label(0.75) == "High"
    assert confidence_label(0.60) == "Moderate"
    assert confidence_label(0.35) == "Low"
    assert confidence_label(0.10) == "Very low"
    assert confidence_label(0.0, has_evidence=False) == "No evidence"


def test_contradicted_evidence_is_not_reported_as_no_evidence(two_source_chunks):
    """Sources that exist but disagree bottom out at 0 - that is 'Very low',
    not 'No evidence', which would misdescribe what the user is looking at."""
    wiped_out = score_confidence(two_source_chunks,
                                 [contradiction("high")] * 3, CITED_ANSWER)
    assert wiped_out["score"] == 0.25
    assert wiped_out["label"] == "Very low"

    assert score_confidence([], [], CITED_ANSWER)["label"] == "No evidence"


def test_vector_similarity_is_preferred_over_the_blended_score():
    """Blended scores rank candidates; vector similarity measures quality.
    Mixing keyword overlap into the confidence figure would deflate it."""
    chunk = make_chunk("evidence", "a.pdf", STRONG)
    chunk["blended_score"] = 0.30      # dragged down by low keyword overlap
    chunk["vector_similarity"] = STRONG

    result = score_confidence([chunk], [], CITED_ANSWER)
    strength = next(f["effect"] for f in result["factors"] if f["label"] == "Evidence strength")
    assert strength == 1.0             # calibrated from STRONG, not from 0.30


def test_build_report_numbers_sources_from_one(two_source_chunks):
    report = build_report("When was the meeting?", CITED_ANSWER, two_source_chunks)

    assert [s["number"] for s in report["sources"]] == [1, 2, 3]
    assert report["sources"][0]["source"] == "witness_a.pdf"
    assert report["confidence"]["label"] == "High"
    assert report["contradictions"] == []
