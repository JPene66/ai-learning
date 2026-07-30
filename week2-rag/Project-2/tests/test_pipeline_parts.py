"""
Tests for the remaining pure pieces of the pipeline: cleaning, chunking,
scoring, re-rank parsing, follow-up rewriting and evaluation parsing.

Like the contradiction tests, anything that would call an LLM gets a fake one
injected, so the whole suite runs offline.
"""

from conftest import make_chunk

from text_cleaner import clean_text, normalize_whitespace, remove_repeated_lines
from ingest import chunk_text, stable_chunk_id
from search import keyword_score, _similarity_from_distance
from rerank import rerank
from multiturn import add_turn, needs_rewrite, rewrite_followup
from rag_eval import normalize_evaluation
from context_assembly import assemble_context, build_prompt


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def test_whitespace_is_normalized():
    assert normalize_whitespace("a    b\n\n\n\nc  ") == "a b\n\nc"


def test_repeated_headers_and_footers_are_removed():
    pages = [
        "ACME CONFIDENTIAL\nIntro text here.\nPage 1",
        "ACME CONFIDENTIAL\nRevenue grew 12%.\nPage 2",
        "ACME CONFIDENTIAL\nOutlook is stable.\nPage 3",
    ]
    cleaned = clean_text("", pages=pages)

    assert "ACME CONFIDENTIAL" not in cleaned
    assert "Page 1" not in cleaned      # "Page N" is matched as one signature
    assert "Revenue grew 12%." in cleaned


def test_real_content_is_not_mistaken_for_boilerplate():
    """Long repeated lines are content, not headers - the length guard matters."""
    sentence = ("This paragraph is long enough that it could not plausibly be a "
                "page header or footer, and must be preserved.")
    pages = [f"Header\n{sentence}\nPage {i}" for i in range(1, 4)]
    cleaned = clean_text("", pages=pages)
    assert sentence in cleaned


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def test_chunks_respect_size_and_drop_empties():
    text = "\n\n".join("word " * 100 for _ in range(6))
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    # allow a little slack for the paragraph-aware joining
    assert all(len(chunk) <= 900 for chunk in chunks)


def test_oversized_paragraph_is_split_with_overlap():
    paragraph = "x" * 2000
    chunks = chunk_text(paragraph, chunk_size=500, overlap=100)
    assert len(chunks) > 1


def test_chunk_ids_are_stable_across_runs():
    """Re-ingesting the same file must produce the same IDs, so upsert updates
    the existing chunks instead of duplicating the document."""
    assert stable_chunk_id("report.pdf", 3) == stable_chunk_id("report.pdf", 3)
    assert stable_chunk_id("report.pdf", 3) != stable_chunk_id("report.pdf", 4)
    assert stable_chunk_id("report.pdf", 3) != stable_chunk_id("other.pdf", 3)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def test_cosine_distance_converts_to_similarity():
    assert _similarity_from_distance(0.0) == 1.0     # identical
    assert _similarity_from_distance(1.0) == 0.0     # unrelated
    assert _similarity_from_distance(2.0) == 0.0     # opposite, clamped not negative
    assert _similarity_from_distance(0.25) == 0.75


def test_keyword_score_ignores_stopwords_and_punctuation():
    text = "The GDPR fine was issued in 2023."

    assert keyword_score(text, "GDPR fine") == 1.0
    assert keyword_score(text, "gdpr, fine!") == 1.0     # punctuation stripped
    assert keyword_score(text, "the and for") == 0.0     # all stopwords
    assert keyword_score(text, "") == 0.0
    assert 0 < keyword_score(text, "GDPR appeal") < 1


# --------------------------------------------------------------------------
# Re-ranking (fake LLM)
# --------------------------------------------------------------------------
CANDIDATES = [make_chunk(f"chunk {i}", f"s{i}.pdf", 0.5) for i in range(4)]


def test_rerank_reorders_by_model_ranking():
    reranked = rerank("q", CANDIDATES, keep_top=2,
                      ask_json_fn=lambda p, **k: {"ranking": [3, 0, 1, 2]})
    assert [c["text"] for c in reranked] == ["chunk 3", "chunk 0"]


def test_rerank_survives_dirty_rankings():
    """String indices, duplicates and out-of-range numbers are all real
    failure modes - none of them should lose a candidate."""
    reranked = rerank("q", CANDIDATES, keep_top=4,
                      ask_json_fn=lambda p, **k: {"ranking": ["2", 2, 99, None, 0]})

    texts = [c["text"] for c in reranked]
    assert texts[:2] == ["chunk 2", "chunk 0"]
    assert len(texts) == 4 and len(set(texts)) == 4   # nothing lost, nothing duplicated


def test_rerank_falls_back_to_search_order_on_failure():
    def broken(prompt, **kwargs):
        raise RuntimeError("API down")

    reranked = rerank("q", CANDIDATES, keep_top=2, ask_json_fn=broken)
    assert [c["text"] for c in reranked] == ["chunk 0", "chunk 1"]


def test_rerank_skips_the_call_for_a_single_candidate():
    def explode(prompt, **kwargs):
        raise AssertionError("should not call the LLM")

    assert rerank("q", CANDIDATES[:1], ask_json_fn=explode) == CANDIDATES[:1]


# --------------------------------------------------------------------------
# Multi-turn
# --------------------------------------------------------------------------
HISTORY = [
    {"role": "user", "content": "What did the witness say about the car?"},
    {"role": "assistant", "content": "A blue sedan leaving at speed [Source 1]."},
]


def test_first_question_is_never_rewritten():
    def explode(prompt, **kwargs):
        raise AssertionError("no history means nothing to rewrite against")

    assert rewrite_followup("What happened?", [], ask_fn=explode) == "What happened?"


def test_short_followup_is_rewritten():
    rewritten = rewrite_followup(
        "And what time was that?", HISTORY,
        ask_fn=lambda p, **k: "What time did the witness see the car?",
    )
    assert rewritten == "What time did the witness see the car?"


def test_long_selfcontained_question_is_left_alone():
    question = ("What did the second witness statement say about the colour of "
                "the vehicle seen leaving the scene that evening?")
    assert not needs_rewrite(question, HISTORY)
    assert rewrite_followup(question, HISTORY, ask_fn=lambda p, **k: "mangled") == question


def test_rewrite_failure_keeps_the_original_question():
    def broken(prompt, **kwargs):
        raise RuntimeError("API down")

    assert rewrite_followup("And then?", HISTORY, ask_fn=broken) == "And then?"


def test_rambling_rewrite_is_rejected():
    """A model that explains itself instead of rewriting shouldn't poison retrieval."""
    essay = "Sure! " + ("The user is asking about the car. " * 20)
    assert rewrite_followup("And then?", HISTORY, ask_fn=lambda p, **k: essay) == "And then?"


def test_add_turn_does_not_mutate_the_caller_list():
    original = []
    updated = add_turn(original, "user", "hello")
    assert original == []
    assert updated == [{"role": "user", "content": "hello"}]


# --------------------------------------------------------------------------
# Evaluation parsing
# --------------------------------------------------------------------------
def test_evaluation_scores_are_normalized_to_0_1():
    result = normalize_evaluation({"faithfulness": 90, "relevance": 8, "groundedness": "0.5"})
    assert result["faithfulness"] == 0.9   # 0-100 scale
    assert result["relevance"] == 0.8      # 0-10 scale
    assert result["groundedness"] == 0.5   # string
    assert result["overall"] == 0.73


def test_missing_metrics_become_none():
    result = normalize_evaluation({"faithfulness": 1.0})
    assert result["relevance"] is None
    assert result["overall"] == 1.0


def test_garbage_evaluation_returns_empty_metrics():
    result = normalize_evaluation("not a dict")
    assert result["overall"] is None
    assert all(result[m] is None for m in ("faithfulness", "relevance", "groundedness"))


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------
def test_context_is_labeled_with_numbered_sources():
    context = assemble_context([
        make_chunk("first", "a.pdf"),
        make_chunk("second", "b.pdf"),
    ])
    assert "[Source 1: a.pdf]" in context
    assert "[Source 2: b.pdf]" in context


def test_prompt_includes_evidence_history_and_citation_rules():
    prompt = build_prompt("When?", [make_chunk("the 3rd", "a.pdf")],
                          conversation_history=HISTORY)

    assert "[Source 1: a.pdf]" in prompt
    assert "the 3rd" in prompt
    assert "blue sedan" in prompt          # history carried through
    assert "[Source 1]" in prompt          # citation format spelled out
    assert "QUESTION: When?" in prompt


def test_prompt_with_no_evidence_asks_for_a_refusal():
    prompt = build_prompt("When?", [])
    assert "no evidence" in prompt.lower()
