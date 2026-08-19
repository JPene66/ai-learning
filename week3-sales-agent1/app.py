"""
app.py
Apex - Autonomous Sales Agent. Streamlit front end.

Run locally with:  streamlit run app.py
Deployment instructions are in README.md.
"""

import uuid
import streamlit as st
from graph import run_turn, get_thread_state

st.set_page_config(page_title="Apex - Autonomous Sales Agent", page_icon="", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "leads_started" not in st.session_state:
    st.session_state.leads_started = []  # list of thread_ids started this session

st.title("Apex: Autonomous Sales Agent")
st.caption("Source → Qualify → Close, with persistent state, guardrails, and human-in-the-loop escalation.")

with st.sidebar:
    st.header("Start a New Lead")
    industry = st.selectbox("Target industry", ["fintech", "healthtech", "logistics", "retail"])
    if st.button("Source a New Lead", type="primary"):
        thread_id = f"lead_{uuid.uuid4().hex[:8]}"
        with st.spinner("Apex is sourcing and qualifying a lead..."):
            run_turn(thread_id, lead_id=thread_id, initial_lead={"industry": industry})
        st.session_state.thread_id = thread_id
        st.session_state.leads_started.append(thread_id)
        st.rerun()

    st.divider()
    st.subheader("Active Leads")
    if st.session_state.leads_started:
        for tid in reversed(st.session_state.leads_started):
            state = get_thread_state(tid)
            if not state:
                continue
            label = f"{state.get('lead', {}).get('company', tid)} — {state.get('deal_stage', '?')}"
            if st.button(label, key=f"select_{tid}", use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()
    else:
        st.write("No leads started yet.")

# ---------------- Main panel ----------------
if not st.session_state.thread_id:
    st.info("Source a new lead from the sidebar to get started.")
    st.stop()

thread_id = st.session_state.thread_id
state = get_thread_state(thread_id)

if not state:
    st.warning("No state found for this thread yet.")
    st.stop()

lead = state.get("lead", {})
deal_stage = state.get("deal_stage", "New")

# --- Pipeline status ---
stage_order = ["New", "Sourced", "Qualified", "Engaging", "Negotiating", "Booked", "Escalated", "Disqualified", "Lost"]
cols = st.columns([2, 2, 2, 2, 1])
cols[0].metric("Company", lead.get("company", "—"))
cols[1].metric("Deal Stage", deal_stage)
cols[2].metric("Segment", state.get("segment", "Unscored"))
cols[3].metric("Qualification Score", f"{state.get('qualification_score', 0)}/100")
cols[4].metric("Est. Cost", f"${state.get('total_estimated_cost_usd', 0):.3f}")

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
        st.warning("Apex needs your approval before continuing.")
        st.write(f"**Proposed action:** {pending.get('action')}")
        st.write(f"**Reasoning:** {pending.get('reasoning')}")
        st.write(f"**Consequence:** {pending.get('consequence')}")
        feedback = st.text_area("Optional note / reply to send as yourself", key=f"feedback_{thread_id}")
        c1, c2 = st.columns(2)
        if c1.button("Approve", type="primary", use_container_width=True):
            with st.spinner("Resuming..."):
                run_turn(thread_id, human_decision="approve", human_feedback=feedback or None)
            st.rerun()
        if c2.button("Reject", use_container_width=True):
            with st.spinner("Resuming..."):
                run_turn(thread_id, human_decision="reject", human_feedback=feedback or None)
            st.rerun()
    elif deal_stage in ("Engaging", "Negotiating"):
        reply = st.chat_input("Type as the lead (simulate their reply)...")
        if reply:
            with st.spinner("Apex is responding..."):
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