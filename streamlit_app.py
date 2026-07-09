import streamlit as st
import os
from core import new_state, generate_stream, load_rag_components, build_graph

st.set_page_config(page_title="Wearable Context Engine", layout="wide")
st.title("IoT Wearable Context Engine — Live Dashboard")

# API key: works both for local/tunnel testing (env var) and Streamlit Cloud (st.secrets)
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

# Cache expensive resources so they load once, not on every interaction
@st.cache_resource
def get_pipeline():
    embedder, reranker, collection = load_rag_components()
    graph = build_graph(embedder, reranker, collection, groq_api_key)
    return graph

app_graph = get_pipeline()

if "log" not in st.session_state:
    st.session_state.log = []

with st.sidebar:
    st.header("Controls")
    scenario = st.selectbox("Simulated scenario", ["normal", "sleeping", "stressed_commuting", "emergency_spike"])
    num_ticks = st.slider("Ticks per round", 2, 8, 4)
    run_button = st.button("Run one round")

if run_button:
    state = new_state()
    state["telemetry_window"] = generate_stream(num_ticks, scenario=scenario)
    with st.spinner("Running agent pipeline..."):
        state = app_graph.invoke(state)
    st.session_state.log.append({
        "scenario": scenario,
        "profile": state["profile_summary"],
        "action": state["last_action"],
        "reasoning": state["last_reasoning"],
        "context": state["retrieved_context"],
    })

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
