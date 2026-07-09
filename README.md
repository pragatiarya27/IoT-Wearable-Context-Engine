# IoT-Wearable-Context-Engine
A LangGraph multi-agent system that acts as the AI brain for a wearable device — ingesting mock smartwatch telemetry and autonomously deciding when to alert the user, using RAG-grounded Groq LLMs.
<div align="center">

# ⌚ IoT Wearable Context Engine


> This project gives a wearable device an autonomous AI brain that *understands* context
> and *decides* when to act, when to nudge, and when to stay silent.

## 🎯 The Problem This Solves

| Without this project | With this project |
|---|---|
| Watch shows HR = 158 bpm. That's it. | Agent detects HR spike + zero movement → triggers emergency protocol |
| No context awareness | Knows user is sleeping → suppresses all non-critical alerts |
| Dumb notifications at any time | User is navigating unfamiliar city → stays completely silent |
| Single sensor reactions | Two-signal escalation rule prevents false alarms |

---

## 🧠 How It Works

```
 📡 Sensor Stream
 ─────────────────────────────────────────────────────────────
  heart_rate │ accelerometer │ gps_speed │ active_app │ battery
 ─────────────────────────────────────────────────────────────
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  🔍  PROFILER AGENT  (LLaMA 3.1 8B · Groq)  │
 │                                              │
 │  Reads rolling window of sensor ticks        │
 │  → "User HR is dangerously high at 158 bpm  │
 │     while completely stationary."            │
 └──────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  📚  RAG PIPELINE  (ChromaDB + Reranker)     │
 │                                              │
 │  Vector search → Cross-encoder rerank        │
 │  → Top 3 protocol rules fetched              │
 │    e.g. "Emergency only when 2 signals       │
 │           agree: high HR + no movement"      │
 └──────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────┐
 │  ⚡  ACTION AGENT  (LLaMA 3.3 70B · Groq)   │
 │                                              │
 │  Profile + raw signals + protocol rules      │
 │  → {"action": "emergency",                   │
 │     "reasoning": "HR 158 + accel 0.02 =      │
 │      two signals confirm cardiac risk"}      │
 └──────────────────────────────────────────────┘
        │
        ▼
  🟢 silent   🟡 notify   🔴 emergency
```

---

## ✨ What Makes This Different

**🔗 Real multi-agent separation** — Profiler Agent only answers *"what is the user doing?"*. Action Agent only answers *"what should we do?"*. They share no code — only a typed state dict.

**📚 RAG-grounded decisions** — the Action Agent doesn't rely on LLM memorised knowledge. It retrieves the relevant health protocol rule from ChromaDB first, then reasons on top of it.

**🎯 Two-stage retrieval** — ChromaDB does fast vector search to get candidates; a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores them for precision. Catches nuance that cosine similarity misses.

**🛡️ Two-signal escalation** — `emergency` only fires when HR spike AND near-zero accelerometer agree. A single noisy reading never triggers a false alarm.

**📊 Built-in eval harness** — 8 labelled test cases with accuracy, emergency recall, and false-positive tracking built into the notebook.

**💸 Entirely free** — local embeddings (no API needed), Groq free tier for both LLMs, ChromaDB on disk, Streamlit Cloud free hosting.

---

## 🗂️ Project Structure

```
📦 wearable-context-engine/
│
├── 📓 IOT_Wearable_Context_Engine.ipynb   ← Everything lives here (36 cells)
│
├── 🧩 core.py                             ← Auto-exported from notebook
│   ├── TelemetryTick + AgentState         │  TypedDicts for the pipeline
│   ├── generate_tick / generate_stream    │  4-scenario mock sensor simulator
│   ├── load_rag_components()              │  Embedder + reranker + ChromaDB
│   ├── retrieve_and_rerank()              │  Two-stage RAG retrieval
│   └── build_graph()                      │  LangGraph graph factory
│
├── 🖥️  streamlit_app.py                   ← Auto-exported from notebook
│
├── 📁 corpus/
│   ├── protocols.md                       ← 5 health decision protocol rules
│   └── baseline_synthetic.json            ← Synthetic 7-day user history
│
├── 📁 vectorstore/                        ← ChromaDB persistent store
│
├── 📋 eval_test_scenarios.json            ← 8 labelled test cases
├── 📊 eval_results.json                   ← Evaluation output
└── 📄 requirements.txt
```

---

## 🎭 Scenarios & Expected Behaviour

| Scenario | HR | Accelerometer | App | What the agent does |
|---|---|---|---|---|
| 😴 `sleeping` | 48–60 bpm | ≈ 0.0 | idle | **Silent** — suppresses all notifications |
| 🚇 `stressed_commuting` | 95–115 bpm | 0.3–0.8 | navigation | **Silent** — user is navigating, don't interrupt |
| 💻 `normal` | 65–85 bpm | 0.1–0.3 | beads_counter / messaging | **Silent** — everything normal |
| 🚨 `emergency_spike` | 140–170 bpm | ≈ 0.0 | idle | **🔴 Emergency** — HR spike + no movement = two signals |

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| 🤖 Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` | Clean node wiring, typed state, extensible |
| 🧠 Profiler LLM | Groq `llama-3.1-8b-instant` | Fast + cheap for one-sentence summarisation |
| ⚡ Action LLM | Groq `llama-3.3-70b-versatile` | Stronger reasoning for structured JSON decisions |
| 🔢 Embeddings | `BAAI/bge-small-en-v1.5` (local) | Fully local — no API key, no cost, no rate limits |
| 🗄️ Vector DB | ChromaDB `PersistentClient` | Persisted to disk, survives notebook restarts |
| 🎯 Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Precision reranking over vector search candidates |
| 🖥️ UI | Streamlit + ngrok | Live dashboard tunnelled from Colab |
| 🧪 Dev environment | Google Colab | Free GPU, zero local setup |

---

## 🚀 Quick Start

### Step 1 — Get your free Groq API key

Go to [console.groq.com](https://console.groq.com) → Sign up → API Keys → **Create Key**

### Step 2 — Open the notebook in Colab

```
Click: Open in Colab → Add secret named Groq_api → paste your key → toggle access ON
```

### Step 3 — Run all cells top to bottom

```
Cell 1   pip install dependencies
Cell 2   Load Groq key from Colab Secrets
Cell 3–5 Define state + simulator, print sample ticks  ✅ verify output
Cell 6–8 Write protocols.md → chunk on ## headers
Cell 9   Embed chunks → load into ChromaDB             ✅ verify "5 chunks added"
Cell 10  Test raw vector retrieval
Cell 11  Load cross-encoder reranker
Cell 12  Define retrieve_and_rerank() (2-stage)
Cell 13  Test reranked retrieval                       ✅ verify top chunk matches
Cell 14  Load Profiler LLM → connectivity test         ✅ verify "OK"
Cell 15  Test profiler_node() on emergency_spike       ✅ verify "dangerously high"
Cell 16  Load Action LLM → connectivity test
Cell 18  Define action_node() with JSON parsing
Cell 19  Full pipeline test — emergency scenario       ✅ verify action = emergency
Cell 20  Full pipeline test — normal scenario          ✅ verify action = silent
Cell 21  Build LangGraph StateGraph
Cell 22  Run graph end-to-end
Cell 24  run_simulation() — 3 rounds
Cell 27  run_evaluation() — all 8 eval cases          ✅ check accuracy + recall
Cell 30  %%writefile core.py
Cell 32  %%writefile streamlit_app.py
Cell 34  Launch Streamlit + ngrok → get public URL    🚀 open in browser
```

> **Tip:** Embeddings and reranking are fully local — no HuggingFace token needed.

---

## 📊 Evaluation Results

The notebook includes a built-in evaluation harness. After running Cell 27–28, you'll see:

```
Case 1 (emergency_spike):  expected=emergency, actual=emergency  ✅
Case 2 (emergency_spike):  expected=emergency, actual=emergency  ✅
Case 3 (normal):           expected=silent,    actual=silent     ✅
Case 4 (normal):           expected=silent,    actual=silent     ✅
Case 5 (sleeping):         expected=silent,    actual=silent     ✅
Case 6 (sleeping):         expected=silent,    actual=silent     ✅
Case 7 (stressed_commuting): expected=silent,  actual=silent     ✅
Case 8 (stressed_commuting): expected=silent,  actual=silent     ✅

Overall accuracy:    8/8 (100%)
Emergency recall:    2/2 (100%)
False emergencies:   0
```

---

## 🖥️ Live Dashboard

After running Cell 34 you get a public Streamlit URL. The dashboard has:

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar           │  Current State                     │
│  ──────────────    │  ──────────────────────────────    │
│  Scenario ▼        │  Profile: "HR dangerously high     │
│  [emergency_spike] │  at 162 bpm, user stationary"      │
│                    │                                    │
│  Ticks per round   │  Action: 🔴 emergency              │
│  [====4====]       │  Reasoning: "Two signals: HR 162   │
│                    │  + accel 0.02 confirm cardiac risk" │
│  [Run one round]   │                                    │
│                    │  Retrieved context:                 │
│                    │  › HR >140 + near-zero accel =      │
│                    │    emergency check-in required      │
├─────────────────────────────────────────────────────────┤
│  Decision Log                                           │
│  ───────────────────────────────────────────────────    │
│  Scenario              │ Action                         │
│  emergency_spike       │ 🔴 emergency                   │
│  normal                │ 🟢 silent                      │
│  sleeping              │ 🟢 silent                      │
└─────────────────────────────────────────────────────────┘
```

> **ngrok note:** If the tunnel fails, go to [dashboard.ngrok.com/endpoints](https://dashboard.ngrok.com/endpoints) → delete the active endpoint → re-run Cell 34.

---

## 🧠 Design Decisions

<details>
<summary><b>Why two LLMs of different sizes?</b></summary>

The Profiler Agent only needs to summarise numbers into one sentence — `llama-3.1-8b-instant` is fast and cheap for this. The Action Agent needs to reason about competing signals, apply protocol rules, and produce structured JSON — `llama-3.3-70b-versatile` gives meaningfully better precision where it counts.

</details>

<details>
<summary><b>Why header-based chunking instead of fixed-size?</b></summary>

Each `##` section in `protocols.md` is one complete decision rule. Fixed-size chunking would split rules mid-sentence, meaning a retrieved chunk might contain half a rule — unusable for grounding a decision. Section-boundary chunking keeps each chunk self-contained.

</details>

<details>
<summary><b>Why a cross-encoder reranker on top of vector search?</b></summary>

Vector search retrieves by embedding similarity, which can surface topically related but contextually wrong chunks. The cross-encoder scores each (query, candidate) pair *together*, catching nuance that cosine similarity misses — especially when the query contains multiple conflicting signals like "high HR AND low battery."

</details>

<details>
<summary><b>Why the two-signal escalation rule?</b></summary>

Wearable sensors are noisy. A single HR spike could be a sensor glitch, a sudden movement, or strong emotion. Requiring two independent signals before triggering `emergency` prevents false alarms that erode user trust — a system that cries wolf gets ignored.

</details>

---

## ⚠️ Groq Free Tier Tips

- Each pipeline run = **2 Groq API calls** (Profiler + Action)
- Free tier: ~14,400 requests/day, 6,000 tokens/minute
- The eval harness has `time.sleep(2.0)` between cases — increase to `3.0` if you hit TPM limits
- Both embedding models run **locally** — zero API calls for RAG

---

## 📄 License

MIT — free to use, modify, and build on.

---

<div align="center">

**Built with** LangGraph · Groq · ChromaDB · sentence-transformers · Streamlit

*If this project helped you, consider giving it a ⭐*

</div>

