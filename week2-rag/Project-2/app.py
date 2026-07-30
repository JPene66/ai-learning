"""
Case File AI - Week 2 project.

A retrieval-augmented research assistant over a "case file": a collection of
PDFs, web articles and text files. It retrieves relevant evidence, generates a
cited answer, flags where sources contradict each other, and scores how much
you should trust the result.

Run with:  streamlit run app.py

This file is deliberately thin. Every pipeline step lives in its own module in
the parent folder (search.py, rerank.py, contradiction.py, report.py...) and
this file only handles the user interface.
"""

import os
import sys
import tempfile

import streamlit as st

# The pipeline modules live one folder up, alongside the rest of the week's
# work. Adding that folder to the import path keeps ONE copy of each step
# instead of duplicating ten files into this project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import MissingAPIKeyError  # noqa: E402
from storing import get_collection, list_sources, delete_source  # noqa: E402
from pipeline import ingest_and_store, answer_question  # noqa: E402
from contradiction import summarize_contradictions  # noqa: E402
from multiturn import add_turn  # noqa: E402

# Where the case file is stored. Override CASEFILE_DB to keep several separate
# case files side by side, or to point the tests at a throwaway database.
CHROMA_PATH = os.environ.get(
    "CASEFILE_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"),
)

st.set_page_config(page_title="Case File AI", page_icon="🗂️", layout="wide")


@st.cache_resource
def load_collection():
    """cache_resource keeps ONE Chroma client alive across Streamlit's reruns.
    Without it, every keystroke would open a new connection to the database."""
    return get_collection(persist_path=CHROMA_PATH)


collection = load_collection()

if "history" not in st.session_state:
    st.session_state.history = []
if "reports" not in st.session_state:
    st.session_state.reports = {}   # question index -> full report dict


# --------------------------------------------------------------------------
# Sidebar: the case file itself, plus retrieval settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🗂️ Case file")

    sources = list_sources(collection)
    if sources:
        st.caption(f"{len(sources)} source(s), {sum(sources.values())} chunks indexed")
        for name, count in sorted(sources.items()):
            row, button = st.columns([5, 1])
            row.write(f"**{os.path.basename(name)}**  \n`{count} chunks`")
            if button.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                delete_source(collection, name)
                st.rerun()
    else:
        st.info("No evidence loaded yet. Add some below.")

    st.divider()
    st.subheader("Add evidence")

    uploads = st.file_uploader(
        "PDF, .txt or .md files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Add files to case", use_container_width=True):
        for upload in uploads:
            with st.spinner(f"Reading {upload.name}..."):
                # document_loader works on file paths, so the upload is written
                # to a temp file first; source_name keeps the ORIGINAL filename
                # as the citable label.
                suffix = os.path.splitext(upload.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(upload.getvalue())
                    tmp_path = tmp.name
                try:
                    result = ingest_and_store(collection, tmp_path, source_name=upload.name)
                    st.success(f"{upload.name}: {result['chunks_stored']} chunks indexed")
                except MissingAPIKeyError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not read {upload.name}: {e}")
                finally:
                    os.unlink(tmp_path)
        st.rerun()

    url = st.text_input("...or a web article URL", placeholder="https://example.com/article")
    if url and st.button("Add URL to case", use_container_width=True):
        with st.spinner(f"Fetching {url}..."):
            try:
                result = ingest_and_store(collection, url)
                st.success(f"{result['chunks_stored']} chunks indexed from the article")
                st.rerun()
            except MissingAPIKeyError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Could not load that URL: {e}")

    st.divider()
    st.subheader("Settings")
    keep_top = st.slider("Sources per answer", 2, 8, 4)
    n_candidates = st.slider("Candidates retrieved before re-ranking", 4, 24, 12)
    detect_conflicts = st.checkbox("Detect contradictions", value=True)
    run_evaluation = st.checkbox("Grade the answer (extra LLM call)", value=False)

    if st.session_state.history and st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.reports = {}
        st.rerun()


# --------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------
def render_confidence(confidence):
    """Confidence isn't just a number - every factor that produced it is shown,
    so a user can disagree with the score for a specific reason."""
    colors = {"High": "🟢", "Moderate": "🟡", "Low": "🟠",
              "Very low": "🔴", "No evidence": "⚪"}
    dot = colors.get(confidence["label"], "⚪")

    st.progress(
        confidence["score"],
        text=f"{dot} Confidence: **{confidence['label']}** ({confidence['percent']}%)",
    )
    with st.expander("Why this score?"):
        for factor in confidence["factors"]:
            effect = factor["effect"]
            sign = "+" if effect > 0 else ""
            marker = "➕" if effect > 0 else ("➖" if effect < 0 else "ℹ️")
            st.markdown(f"{marker} **{factor['label']}** ({sign}{effect:.2f}) — {factor['detail']}")


def render_contradictions(contradictions):
    if not contradictions:
        st.caption("✅ " + summarize_contradictions(contradictions))
        return

    st.warning("⚠️ " + summarize_contradictions(contradictions))
    for c in contradictions:
        with st.expander(f"[{c['severity'].upper()}] {c['explanation']}"):
            st.markdown(f"**{os.path.basename(c['source_a'])} says:** {c['claim_a']}")
            st.markdown(f"**{os.path.basename(c['source_b'])} says:** {c['claim_b']}")


def render_sources(sources):
    if not sources:
        return
    st.markdown("**Sources used**")
    for source in sources:
        label = (f"[Source {source['number']}] {os.path.basename(source['source'])} "
                 f"— relevance {source['score']:.0%}")
        with st.expander(label):
            st.write(source["text"])


def render_evaluation(evaluation):
    if not evaluation or evaluation.get("overall") is None:
        return
    st.markdown("**Self-evaluation**")
    columns = st.columns(4)
    for column, metric in zip(
        columns, ("faithfulness", "relevance", "groundedness", "overall")
    ):
        value = evaluation.get(metric)
        column.metric(metric.capitalize(), f"{value:.2f}" if value is not None else "—")
    if evaluation.get("notes"):
        st.caption(evaluation["notes"])


def render_report(report):
    st.markdown(report["answer"])

    if report.get("retrieval_query") and report["retrieval_query"] != report["question"]:
        st.caption(f"🔁 Follow-up rewritten for retrieval: *{report['retrieval_query']}*")

    st.divider()
    render_confidence(report["confidence"])
    render_contradictions(report["contradictions"])
    render_sources(report["sources"])
    render_evaluation(report.get("evaluation"))


# --------------------------------------------------------------------------
# Main panel: the conversation
# --------------------------------------------------------------------------
st.title("🗂️ Case File AI")
st.caption("Ask questions across your evidence. Answers are cited, conflicts are "
           "flagged, and every confidence score is explained.")

for i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and i in st.session_state.reports:
            render_report(st.session_state.reports[i])
        else:
            st.markdown(turn["content"])

question = st.chat_input(
    "Ask about the case file..." if sources else "Add evidence first, then ask a question"
)

if question:
    if not sources:
        st.warning("Add at least one document to the case file before asking a question.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        # History is passed BEFORE this question is added, so the follow-up
        # rewriter sees prior turns without the question it's rewriting.
        history_before = list(st.session_state.history)
        st.session_state.history = add_turn(st.session_state.history, "user", question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the case file..."):
                try:
                    report = answer_question(
                        collection,
                        question,
                        history=history_before,
                        n_candidates=n_candidates,
                        keep_top=keep_top,
                        detect_conflicts=detect_conflicts,
                        evaluate=run_evaluation,
                    )
                except MissingAPIKeyError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Something went wrong answering that: {e}")
                    st.stop()

            render_report(report)

        st.session_state.history = add_turn(
            st.session_state.history, "assistant", report["answer"]
        )
        st.session_state.reports[len(st.session_state.history) - 1] = report
