"""
app.py
The Persuasion Arena - Streamlit front end.

Run locally with:  streamlit run app.py
Deployment instructions are in README.md.
"""

from datetime import datetime
import streamlit as st
from persona import PERSONAS
from arena import run_multi_persona_round

st.set_page_config(page_title="The Persuasion Arena", page_icon="", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("The Persuasion Arena")
st.caption("Debate AI-powered negotiation opponents and get your argument scored by an AI judge.")

with st.sidebar:
    st.header("Round Setup")
    selected = st.multiselect(
        "Choose your opponents",
        options=list(PERSONAS.keys()),
        format_func=lambda k: PERSONAS[k]["label"],
        default=["diplomat", "aggressive"],
    )
    version = st.radio(
        "Opponent prompt version",
        ["v1", "v2"],
        horizontal=True,
        help="Compare two different system-prompt versions for the same persona (prompt A/B testing).",
    )
    st.divider()
    st.subheader("Session Leaderboard")
    if st.session_state.history:
        avg = sum(r["score"]["average"] for r in st.session_state.history) / len(st.session_state.history)
        best = max(st.session_state.history, key=lambda r: r["score"]["average"])
        st.metric("Rounds played", len(st.session_state.history))
        st.metric("Average opponent score", round(avg, 2))
        st.metric("Best single reply", f"{best['score']['average']} ({best['persona_label']})")
    else:
        st.write("No rounds played yet.")

offer = st.text_area(
    "Your offer / argument",
    placeholder="e.g. I'd like a 15% discount in exchange for a 12-month contract.",
)

submit_disabled = not offer or not selected
if st.button("Submit to Arena", type="primary", disabled=submit_disabled):
    with st.spinner("Opponents are responding..."):
        results = run_multi_persona_round(selected, version, offer)
    st.session_state.history.extend(results)

    for r in results:
        with st.container(border=True):
            st.subheader(r["persona_label"])
            st.write(r["reply"])
            score = r["score"]
            cols = st.columns(5)
            cols[0].metric("Logic", score["logic"])
            cols[1].metric("Evidence", score["evidence"])
            cols[2].metric("Clarity", score["clarity"])
            cols[3].metric("Persuasiveness", score["persuasiveness"])
            cols[4].metric("Average", score["average"])
            if score.get("rationale"):
                st.caption(f"Judge's rationale: {score['rationale']}")

st.divider()
st.subheader("Session History & Export")

if st.session_state.history:
    for i, r in enumerate(st.session_state.history, start=1):
        st.write(f"**Round {i} — {r['persona_label']} ({r['version']})** — average score: {r['score']['average']}")

    report_lines = [
        "# The Persuasion Arena — Session Report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for i, r in enumerate(st.session_state.history, start=1):
        s = r["score"]
        report_lines += [
            f"## Round {i}: {r['persona_label']} ({r['version']})",
            f"**Offer:** {r['offer']}",
            f"**Reply:** {r['reply']}",
            f"**Scores:** Logic {s['logic']}, Evidence {s['evidence']}, Clarity {s['clarity']}, "
            f"Persuasiveness {s['persuasiveness']}, Average {s['average']}",
            f"**Judge rationale:** {s.get('rationale', '')}",
            "",
        ]
    report_text = "\n".join(report_lines)
    st.download_button(
        "Download performance report (Markdown)",
        report_text,
        file_name="persuasion_arena_report.md",
    )

    if st.button("Clear session history"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("Play a round above to build your session history.")