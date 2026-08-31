"""
app.py
Apex - Autonomous Sales Agent. Streamlit front end.

Run locally with:  streamlit run app.py
Deployment instructions are in README.md.
"""

import uuid
import streamlit as st
from graph import run_turn, get_thread_state
from demographics import COUNTRIES, REGIONS, INDUSTRIES

st.set_page_config(page_title="Apex - Autonomous Sales Agent", page_icon="🎯", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "leads_started" not in st.session_state:
    st.session_state.leads_started = []  # list of thread_ids started this session
if "demographics" not in st.session_state:
    st.session_state.demographics = []  # list of {industry, country, region, max_leads} dicts

st.title("Apex Autonomous Sales Agent")
st.caption("Source → Qualify → Close, with persistent state, guardrails, and human-in-the-loop escalation.")

# ────────────────────────────────────────────────────────────────────────────
# Sidebar — Target Demographics
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Target Demographics")
    st.caption("Select a country, region and sector — Apex will source real leads from the web.")

    # ── Step 1: Country — OUTSIDE the form so changing it re-runs the page
    # and immediately refreshes the region list below. ────────────────────────
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = "Nigeria"

    selected_country = st.selectbox(
        "Country",
        options=COUNTRIES,
        index=COUNTRIES.index(st.session_state.selected_country),
        key="live_country",
    )
    # Persist so the form below can read it
    st.session_state.selected_country = selected_country

    # ── Steps 2-4 + submit stay inside the form ──────────────────────────────
    with st.form("add_demographic_form", clear_on_submit=True):
        # ── Step 2: Region — options are built from the live country selection
        country_regions = REGIONS.get(selected_country, [])
        selected_region = st.selectbox(
            "Region / State / City",
            options=["(Whole country)"] + country_regions,
        )
        # Normalise "whole country" to empty string
        region_value = "" if selected_region == "(Whole country)" else selected_region

        # ── Step 3: Industry ─────────────────────────────────────────────────
        selected_industry = st.selectbox(
            "Industry / Sector",
            options=INDUSTRIES,
        )

        # ── Step 4: Number of leads ──────────────────────────────────────────
        max_leads = st.slider("Leads to source", min_value=1, max_value=5, value=2)

        added = st.form_submit_button("Add Target Demographic", use_container_width=True)
        if added:
            st.session_state.demographics.append({
                "industry": selected_industry,
                "country":  selected_country,
                "region":   region_value,
                "max_leads": max_leads,
            })

    # ── Current target list ──────────────────────────────────────────────────
    if st.session_state.demographics:
        st.write("**Current targets:**")
        for i, d in enumerate(st.session_state.demographics):
            loc = d["region"] + ", " + d["country"] if d["region"] else d["country"]
            label = f"{d['industry']} — {loc} ({d['max_leads']} lead{'s' if d['max_leads'] != 1 else ''})"
            c1, c2 = st.columns([4, 1])
            c1.write(f"• {label}")
            if c2.button("✕", key=f"remove_demo_{i}"):
                st.session_state.demographics.pop(i)
                st.rerun()
    else:
        st.info("Add at least one demographic target to begin sourcing.")

    st.divider()

    # ── Single-target sourcing ───────────────────────────────────────────────
    demo_count = len(st.session_state.demographics)

    def _demo_label(d: dict) -> str:
        loc = (d["region"] + ", " if d["region"] else "") + d["country"]
        return f"{d['industry']} — {loc}"

    single_target = st.selectbox(
        "Source from one target",
        options=list(range(demo_count)),
        format_func=lambda i: _demo_label(st.session_state.demographics[i]),
        disabled=demo_count == 0,
    ) if demo_count else None

    if st.button("Source From Selected Target", type="primary", disabled=demo_count == 0):
        target = st.session_state.demographics[single_target]
        loc = (target["region"] + ", " if target["region"] else "") + target["country"]
        thread_id = f"lead_{uuid.uuid4().hex[:8]}"
        with st.spinner(f"Apex is sourcing {target['max_leads']} real {target['industry']} lead(s) in {loc}…"):
            run_turn(thread_id, lead_id=thread_id, initial_lead=target)
        st.session_state.thread_id = thread_id
        st.session_state.leads_started.append(thread_id)
        st.rerun()

    if st.button("Source One Lead From Every Target", disabled=demo_count < 2):
        new_threads = []
        progress = st.progress(0.0, text="Sourcing across all demographics…")
        for i, target in enumerate(st.session_state.demographics):
            thread_id = f"lead_{uuid.uuid4().hex[:8]}"
            run_turn(thread_id, lead_id=thread_id, initial_lead=target)
            new_threads.append(thread_id)
            progress.progress((i + 1) / demo_count, text=f"Sourced from {target['industry']}…")
        st.session_state.leads_started.extend(new_threads)
        st.session_state.thread_id = new_threads[-1]
        st.rerun()
    st.caption(
        "Each run uses a real DuckDuckGo search (free) + a GPT call to parse leads. "
        "'Source Every Target' fires one search per saved target in a single pass."
    )

    st.divider()

    # ── Active leads list ────────────────────────────────────────────────────
    st.subheader("Active Leads")
    if st.session_state.leads_started:
        for tid in reversed(st.session_state.leads_started):
            state = get_thread_state(tid)
            if not state:
                continue
            company = state.get("lead", {}).get("company", tid)
            stage   = state.get("deal_stage", "?")
            country = state.get("country", "")
            region  = state.get("region", "")
            loc_tag = f" ({region + ', ' if region else ''}{country})" if country else ""
            label = f"{company}{loc_tag} — {stage}"
            if st.button(label, key=f"select_{tid}", use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()
    else:
        st.write("No leads started yet.")


# ────────────────────────────────────────────────────────────────────────────
# Main panel
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.thread_id:
    st.info("Source a new lead from the sidebar to get started.")
    st.stop()

thread_id = st.session_state.thread_id
state = get_thread_state(thread_id)

if not state:
    st.warning("No state found for this thread yet.")
    st.stop()

lead       = state.get("lead", {})
deal_stage = state.get("deal_stage", "New")
country    = state.get("country", "")
region     = state.get("region", "")

# ── Pipeline status ──────────────────────────────────────────────────────────
loc_display = (region + ", " if region else "") + country if country else "—"
cols = st.columns([2, 2, 2, 2, 1, 1])
cols[0].metric("Company",           lead.get("company", "—"))
cols[1].metric("Deal Stage",        deal_stage)
cols[2].metric("Location",          loc_display)
cols[3].metric("Segment",           state.get("segment", "Unscored"))
cols[4].metric("Qual. Score",       f"{state.get('qualification_score', 0)}/100")
cols[5].metric("Est. Cost",         f"${state.get('total_estimated_cost_usd', 0):.3f}")

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Conversation")
    history = state.get("conversation_history", [])
    if not history:
        st.write("No conversation yet.")
    for msg in history:
        role = msg["role"]
        if role == "apex":
            st.chat_message("assistant", avatar="🎯").write(msg["content"])
        elif role == "lead":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("user", avatar="🧑‍💼").write(f"**[Human]** {msg['content']}")

    # --- HITL approval panel takes priority over normal chat input ---
    if state.get("needs_human_approval"):
        pending = state.get("pending_action", {})
        st.warning(" Apex needs your approval before continuing.")
        st.write(f"**Proposed action:** {pending.get('action')}")
        st.write(f"**Reasoning:** {pending.get('reasoning')}")
        st.write(f"**Consequence:** {pending.get('consequence')}")
        feedback = st.text_area("Optional note / reply to send as yourself", key=f"feedback_{thread_id}")
        c1, c2 = st.columns(2)
        if c1.button("Approve", type="primary", use_container_width=True):
            with st.spinner("Resuming…"):
                run_turn(thread_id, human_decision="approve", human_feedback=feedback or None)
            st.rerun()
        if c2.button("❌ Reject", use_container_width=True):
            with st.spinner("Resuming…"):
                run_turn(thread_id, human_decision="reject", human_feedback=feedback or None)
            st.rerun()
    elif deal_stage in ("Engaging", "Negotiating"):
        reply = st.chat_input("Type as the lead (simulate their reply)…")
        if reply:
            with st.spinner("Apex is responding…"):
                run_turn(thread_id, latest_lead_message=reply)
            st.rerun()
    elif deal_stage == "Booked":
        st.success("Meeting booked! This lead has been successfully closed.")
    elif deal_stage in ("Disqualified", "Lost"):
        st.error(f"This lead ended as: {deal_stage}")

with right:
    st.subheader("Qualification Detail")
    bant = state.get("bant_medicc", {})
    if bant.get("scores"):
        for k, v in bant["scores"].items():
            st.write(f"**{k.replace('_', ' ').title()}:** {v}/5")
        st.caption(bant.get("rationale", ""))
    else:
        st.write("Not yet scored.")

    st.subheader("Lead Data")
    st.json(lead, expanded=False)

    if state.get("objection_log"):
        st.subheader("Objections Handled")
        for obj in state["objection_log"]:
            with st.expander(obj.get("topic", "objection")):
                st.write(f"**Message:** {obj.get('message')}")
                st.write(f"**Resolution:** {obj.get('resolution')}")

    if state.get("negotiation_offer_pct"):
        st.metric("Current Discount Offered", f"{state['negotiation_offer_pct']}%")

    if state.get("error_log"):
        st.subheader("Errors / Fallbacks")
        for err in state["error_log"]:
            st.caption(err)

st.divider()
st.caption(
    "Built as Project 3 of the 6-Week AI Career-Ready Training Program, applying agent architecture, "
    "memory, guardrails, human-in-the-loop, and advanced retrieval end to end."
)