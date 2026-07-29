"""
app.py
Case File AI - Streamlit front end.

Run locally with:  streamlit run app.py
Deployment instructions are in README.md.
"""

import os
import tempfile
import streamlit as st
from vectorstore import get_collection, add_source_to_case, hybrid_search
from contradiction import find_contradictions
from report import generate_case_report

st.set_page_config(page_title="Case File AI", page_icon="🕵️", layout="wide")

if "collection" not in st.session_state:
    st.session_state.collection = get_collection()
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = []
if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

SAMPLE_CASE_DIR = os.path.join(os.path.dirname(__file__), "sample_case")

st.title("Case File AI")
st.caption("A multi-source investigation assistant — ingest evidence, ask questions, and let it flag contradictions between sources.")

with st.sidebar:
    st.header("Case Evidence")

    if st.button("Load Demo Case: \"The Warehouse 12 Fire\"", type="primary",
                 disabled=st.session_state.demo_loaded):
        with st.spinner("Ingesting demo evidence..."):
            total = 0
            for fname in sorted(os.listdir(SAMPLE_CASE_DIR)):
                fpath = os.path.join(SAMPLE_CASE_DIR, fname)
                count = add_source_to_case(st.session_state.collection, fpath)
                st.session_state.loaded_sources.append(fname)
                total += count
        st.session_state.demo_loaded = True
        st.success(f"Loaded {len(os.listdir(SAMPLE_CASE_DIR))} demo sources ({total} chunks).")

    st.divider()
    st.subheader("Add Your Own Evidence")

    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("Add uploaded file to case"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            count = add_source_to_case(st.session_state.collection, tmp_path)
        st.session_state.loaded_sources.append(uploaded_file.name)
        st.success(f"Added {uploaded_file.name} ({count} chunks).")

    url_input = st.text_input("Or paste a web article URL")
    if url_input and st.button("Add URL to case"):
        with st.spinner(f"Fetching {url_input}..."):
            count = add_source_to_case(st.session_state.collection, url_input)
        st.session_state.loaded_sources.append(url_input)
        st.success(f"Added URL ({count} chunks).")

    st.divider()
    st.subheader("Sources in This Case")
    if st.session_state.loaded_sources:
        for s in st.session_state.loaded_sources:
            st.write(f" {os.path.basename(s)}")
    else:
        st.write("No evidence loaded yet.")

st.subheader("Open an Investigation")
question = st.text_area(
    "What are you investigating?",
    placeholder="e.g. What caused the fire, and what time did it start?",
)

if st.button("Investigate", type="primary", disabled=not question or not st.session_state.loaded_sources):
    with st.spinner("Reviewing evidence..."):
        retrieved = hybrid_search(st.session_state.collection, question, n_results=6)
        contradictions = find_contradictions(retrieved)
        case_report = generate_case_report(question, retrieved, contradictions)

    conf = case_report["confidence"]
    conf_color = "🟢" if conf >= 70 else "🟡" if conf >= 40 else "🔴"

    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Case Report")
    with col2:
        st.metric("Confidence", f"{conf_color} {conf}/100")

    st.write(case_report["answer"])

    if case_report["contradictions"]:
        st.subheader(f"Contradictions Found ({len(case_report['contradictions'])})")
        for c in case_report["contradictions"]:
            severity_icon = {"Low": "🟡", "Medium": "🟠", "High": "🔴"}.get(c["severity"], "🟡")
            with st.container(border=True):
                st.markdown(f"**{severity_icon} {c['topic']}** — severity: {c['severity']}")
                cols = st.columns(2)
                cols[0].markdown(f"**{os.path.basename(c['source_a'])}:**\n\n{c['claim_a']}")
                cols[1].markdown(f"**{os.path.basename(c['source_b'])}:**\n\n{c['claim_b']}")
    else:
        st.success("No contradictions detected between the retrieved sources.")

    with st.expander(f"Evidence Reviewed ({len(retrieved)} chunks)"):
        for r in retrieved:
            st.markdown(
                f"**{os.path.basename(r['source'])}** "
                f"(vector: {r['vector_similarity']}, keyword: {r['keyword_score']}, blended: {r['blended_score']})"
            )
            st.caption(r["text"])
            st.divider()

st.divider()
st.caption(
    "Built as Project 2 of the 6-Week AI Career-Ready Training Program, applying RAG end to end: "
    "ingestion, hybrid search, citation, contradiction detection, and confidence scoring."
)