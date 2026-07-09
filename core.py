import os
import json
import random
from datetime import datetime, timezone
from typing import TypedDict, List, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# ---------- State ----------
class TelemetryTick(TypedDict):
    timestamp: str
    heart_rate: int
    accelerometer: float
    gps_speed: float
    active_app: str
    battery: int

class AgentState(TypedDict):
    telemetry_window: List[TelemetryTick]
    profile_summary: Optional[str]
    retrieved_context: Optional[List[str]]
    last_action: Optional[str]
    last_reasoning: Optional[str]

def new_state() -> AgentState:
    return {
        "telemetry_window": [],
        "profile_summary": None,
        "retrieved_context": None,
        "last_action": None,
        "last_reasoning": None,
    }

# ---------- Simulator ----------
def generate_tick(scenario: str = "normal") -> TelemetryTick:
    now = datetime.now(timezone.utc).isoformat()
    if scenario == "sleeping":
        hr, accel, speed, app = random.randint(48, 60), round(random.uniform(0.0, 0.1), 2), 0.0, "idle"
    elif scenario == "stressed_commuting":
        hr, accel, speed, app = random.randint(95, 115), round(random.uniform(0.3, 0.8), 2), round(random.uniform(3, 15), 1), "navigation"
    elif scenario == "emergency_spike":
        hr, accel, speed, app = random.randint(140, 170), round(random.uniform(0.0, 0.05), 2), 0.0, "idle"
    else:
        hr, accel, speed, app = random.randint(65, 85), round(random.uniform(0.1, 0.3), 2), round(random.uniform(0, 5), 1), random.choice(["beads_counter", "idle", "messaging"])
    return {"timestamp": now, "heart_rate": hr, "accelerometer": accel, "gps_speed": speed, "active_app": app, "battery": random.randint(15, 100)}

def generate_stream(n: int, scenario: str = "normal") -> List[TelemetryTick]:
    return [generate_tick(scenario) for _ in range(n)]

# ---------- RAG components (loaded once, cached by Streamlit) ----------
def load_rag_components():
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    chroma_client = chromadb.PersistentClient(path="./vectorstore")
    collection = chroma_client.get_or_create_collection(name="protocols")
    return embedder, reranker, collection

def retrieve_and_rerank(query, embedder, reranker, collection, top_k_retrieve=5, top_k_final=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k_retrieve)
    candidates = results["documents"][0]
    if not candidates:
        return []
    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k_final]]

# ---------- Prompts ----------
PROFILER_SYSTEM_PROMPT = """You are the Profiler Agent for a wearable device.
Given a rolling window of telemetry readings, summarize the user's current
state in ONE short sentence.

Use these reference ranges when describing heart rate:
- Below 60 bpm: low/resting
- 60-100 bpm: normal
- 100-140 bpm: elevated
- Above 140 bpm: dangerously high

Always explicitly state if heart rate is "dangerously high" when any reading
exceeds 140 bpm, especially if accelerometer is near 0. Do not soften or
downplay high readings. Do not make medical diagnoses, but be direct and
factual about numbers outside normal ranges."""

ACTION_SYSTEM_PROMPT = """You are the Action Agent for a wearable device.
You decide whether to stay silent, notify the user, or trigger an emergency
protocol, based on the user's current profile and relevant protocol rules.

Respond with ONLY a JSON object in this exact format, no other text:
{"action": "silent" | "notify" | "emergency", "reasoning": "one sentence explaining why"}

Follow the escalation rule: only choose "emergency" if at least two signals
in the profile agree on danger (e.g. high heart rate AND no movement)."""

# ---------- Graph builder ----------
def build_graph(embedder, reranker, collection, groq_api_key):
    profiler_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)
    action_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)

    def profiler_node(state: AgentState) -> AgentState:
        window = state["telemetry_window"]
        if not window:
            state["profile_summary"] = "No telemetry data available."
            return state
        readings_text = "\n".join(
            f"HR:{t['heart_rate']} Accel:{t['accelerometer']} Speed:{t['gps_speed']} App:{t['active_app']} Batt:{t['battery']}%"
            for t in window
        )
        messages = [SystemMessage(content=PROFILER_SYSTEM_PROMPT), HumanMessage(content=f"Recent readings:\n{readings_text}")]
        response = profiler_llm.invoke(messages)
        state["profile_summary"] = response.content.strip()
        return state

    def action_node(state: AgentState) -> AgentState:
        profile = state["profile_summary"] or "No profile available."
        window = state["telemetry_window"]
        max_hr = max(t["heart_rate"] for t in window) if window else 0
        min_accel = min(t["accelerometer"] for t in window) if window else 0
        min_battery = min(t["battery"] for t in window) if window else 100

        context_chunks = retrieve_and_rerank(profile, embedder, reranker, collection)
        state["retrieved_context"] = context_chunks
        context_text = "\n\n".join(context_chunks)

        user_message = f"""User profile (narrative): {profile}

Raw computed signals (trust these numbers over the narrative if they conflict):
- Max heart rate in window: {max_hr} bpm
- Min accelerometer in window: {min_accel}
- Min battery in window: {min_battery}%

Relevant protocol rules:
{context_text}

Decide the action."""

        messages = [SystemMessage(content=ACTION_SYSTEM_PROMPT), HumanMessage(content=user_message)]
        response = action_llm.invoke(messages)
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(raw)
            action = parsed.get("action", "silent")
            reasoning = parsed.get("reasoning", "No reasoning provided.")
        except json.JSONDecodeError:
            action, reasoning = "silent", f"Could not parse model output: {raw}"

        state["last_action"] = action
        state["last_reasoning"] = reasoning
        return state

    graph = StateGraph(AgentState)
    graph.add_node("profiler", profiler_node)
    graph.add_node("action", action_node)
    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "action")
    graph.add_edge("action", END)
    return graph.compile()
