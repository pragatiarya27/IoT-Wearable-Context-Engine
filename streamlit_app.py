%%writefile streamlit_app.py
import streamlit as st
import os
import time
from core import new_state, generate_stream, load_rag_components, build_graph

st.set_page_config(page_title="Wearable Context Engine", layout="wide")
st.title("IoT Wearable Context Engine — Live Dashboard")

groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

@st.cache_resource
def get_pipeline():
    embedder, reranker, collection = load_rag_components()
    graph = build_graph(embedder, reranker, collection, groq_api_key)
    return graph

app_graph = get_pipeline()

if "log" not in st.session_state:
    st.session_state.log = []
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = 0

with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox(
        "Simulated scenario",
        ["normal", "sleeping", "stressed_commuting", "emergency_spike"]
    )
    num_ticks = st.slider("Ticks per round", 2, 8, 4)
    run_button = st.button("Run one round")

    # Show cooldown timer
    elapsed = time.time() - st.session_state.last_run_time
    cooldown = 15  # seconds between runs
    if elapsed < cooldown:
        remaining = int(cooldown - elapsed)
        st.warning(f"⏳ Wait {remaining}s before next run (rate limit protection)")
        run_button = False  # block the button

if run_button:
    elapsed = time.time() - st.session_state.last_run_time
    if elapsed < 15:
        st.warning("Please wait before running again — Groq rate limit protection.")
    else:
        state = new_state()
        state["telemetry_window"] = generate_stream(num_ticks, scenario=scenario)

        with st.spinner("Running agent pipeline..."):
            try:
                state = app_graph.invoke(state)
                st.session_state.last_run_time = time.time()
                st.session_state.log.append({
                    "scenario": scenario,
                    "profile": state["profile_summary"],
                    "action": state["last_action"],
                    "reasoning": state["last_reasoning"],
                    "context": state["retrieved_context"],
                })
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    st.error("⚠️ Groq rate limit hit. Please wait 60 seconds and try again.")
                    st.info("Tip: Groq free tier allows ~30 requests/minute. "
                            "Each run uses 2 requests (Profiler + Action Agent).")
                else:
                    st.error(f"Pipeline error: {str(e)}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current State")
    if st.session_state.log:
        latest = st.session_state.log[-1]
        st.write("**Profile summary:**", latest["profile"])
        action_color = {"silent": "🟢", "notify": "🟡", "emergency": "🔴"}
        st.write(f"**Action:** {action_color.get(latest['action'], '')} {latest['action']}")
        st.write("**Reasoning:**", latest["reasoning"])
        st.write("**Retrieved context:**")
        for chunk in latest["context"] or []:
            st.caption(chunk[:200])
    else:
        st.info("Click 'Run one round' to start.")

with col2:
    st.subheader("Decision Log")
    if st.session_state.log:
        st.table([
            {"Scenario": r["scenario"], "Action": r["action"]}
            for r in st.session_state.log
        ])

# Rate limit info at bottom
st.divider()
st.caption("ℹ️ Groq free tier: 30 requests/minute. Each run = 2 requests. "
           "Wait 15s between runs to stay within limits.")
!git add streamlit_app.py
!git commit -m "fix: add rate limit protection and cooldown timer"
!git push origin main
