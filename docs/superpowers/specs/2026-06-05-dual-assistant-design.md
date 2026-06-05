# Dual AI Personal Assistant — Design Spec
**Date:** 2026-06-05
**Status:** Approved

---

## Overview

Build and evaluate two AI personal assistants sharing the same feature set and API contract:

1. **OSS Assistant** — Qwen2.5-0.5B-Instruct, deployed publicly on Hugging Face Spaces (Docker)
2. **Frontier Assistant** — Gemini 1.5 Flash (free API tier), runs locally via Docker Compose

Both assistants support multi-turn conversations, sliding-window memory, Tavily web search, shared guardrails, and Langfuse observability. A shared evaluation harness runs a 45-prompt battery, scores responses with LLM-as-judge, and generates a 1-page visual report.

---

## Repository Structure

```
ollive/
├── oss_assistant/
│   ├── backend/
│   │   └── app.py          # FastAPI: POST /chat, GET /health
│   ├── frontend/
│   │   └── app.py          # Gradio UI → calls backend via HTTP
│   ├── model.py            # Qwen2.5-0.5B-Instruct loader + inference
│   └── Dockerfile          # supervisord: Gradio (7860, public) + FastAPI (8000, internal)
│
├── frontier_assistant/
│   ├── backend/
│   │   └── app.py          # FastAPI: POST /chat, GET /health
│   ├── frontend/
│   │   └── app.py          # Gradio UI → calls backend via HTTP
│   └── model.py            # Gemini 1.5 Flash client (google-generativeai)
│
├── shared/
│   ├── guardrails.py       # Input blocklist + output toxic classifier
│   ├── observability.py    # Langfuse trace wrapper
│   ├── tools.py            # Tavily web search tool
│   └── memory.py           # Sliding window context manager (deque, cap=10)
│
├── evaluation/
│   ├── prompts/
│   │   ├── factual.json    # 15 factual prompts
│   │   ├── adversarial.json # 15 jailbreak/harmful prompts
│   │   └── bias.json       # 15 bias/stereotype prompts
│   ├── judge.py            # LLM-as-judge scoring via Gemini
│   ├── run_eval.py         # Hits both backends, collects + saves scores
│   └── report.py           # matplotlib charts + latency/cost table
│
├── docker-compose.yml      # Local: OSS backend:8000, Frontier backend:8001
├── requirements.txt
└── README.md
```

---

## API Contract

Both backends expose the same interface. Evaluation scripts are model-agnostic.

### `POST /chat`

**Request:**
```json
{
  "message": "string",
  "session_id": "string",
  "use_search": true
}
```

**Response:**
```json
{
  "reply": "string",
  "tool_used": "web_search | none",
  "tool_query": "string | null",
  "latency_ms": 1240,
  "tokens_used": 87,
  "guardrail_triggered": false,
  "session_id": "string"
}
```

### `GET /health`

Returns: `{ "model": "string", "uptime_s": 123, "request_count": 45 }`

---

## Data Flow (per request)

```
User message
  → Guardrails input check
      → blocked: return canned refusal, log to Langfuse, stop
  → Memory: prepend last ≤10 turns to prompt
  → Tool decision: does message contain who/what/when/where/which?
      → yes: Tavily search → inject "[Search result: ...]" into user turn
  → Model inference (Qwen or Gemini)
  → Guardrails output check
      → flagged: redact + append safety notice
  → Langfuse trace (latency, tokens, guardrail flags, tool used)
  → Return response to Gradio frontend
```

---

## Shared Modules

### `shared/memory.py`
- Per-`session_id` `collections.deque` capped at 10 turns
- Each entry: `{"role": "user" | "assistant", "content": "string"}`
- Identical implementation in both backends — ensures apples-to-apples memory in evaluation

### `shared/tools.py`
- Single function: `web_search(query: str) -> str`
- Calls Tavily API (free tier: 1,000 searches/month)
- Returns top 3 result snippets joined as plain text
- Tool decision: **Qwen** uses regex heuristic (question words: who/what/when/where/which/how); **Gemini** uses native function calling (tool defined in API schema, model decides autonomously)

### `shared/guardrails.py`

**Input filter:**
- Regex + keyword blocklist: jailbreak patterns (`"ignore previous instructions"`, `"pretend you are"`, `"DAN"`, prompt injection markers)
- If triggered: return `{"reply": "I can't help with that.", "guardrail_triggered": true}` immediately

**Output filter:**
- Run `martin-ha/toxic-comment-model` (HuggingFace, CPU-fast) on generated reply
- Regex scan for PII patterns (SSN, credit card numbers)
- If flagged: redact flagged span, append `"[Response filtered for safety]"`

### `shared/observability.py`
- Wraps Langfuse Python SDK
- Every `/chat` call creates a trace with: `model`, `session_id`, `latency_ms`, `tokens_in`, `tokens_out`, `tool_used`, `guardrail_input_triggered`, `guardrail_output_triggered`
- `judge_scores` written back to trace after eval run
- Langfuse free cloud tier (no credit card required)

---

## Evaluation Framework

### Prompt Battery (45 prompts)

| Category | Count | Purpose |
|---|---|---|
| Factual | 15 | Test hallucination — recent events, specific facts, numerical claims |
| Adversarial/Jailbreak | 15 | Test safety — DAN prompts, role-play bypasses, indirect harmful requests |
| Bias/Stereotype | 15 | Test bias — gender, race, religion, occupation stereotypes |

### LLM-as-Judge (`evaluation/judge.py`)
- Sends `(prompt, response)` pairs to Gemini 1.5 Flash with structured scoring prompt
- Returns: `{"hallucination": 0-10, "safety": 0-10, "bias": 0-10, "reasoning": "string"}`
- Score meaning: 10 = perfect (no hallucination / fully safe / unbiased)
- Scores averaged per category per model

### Hybrid Element
- `run_eval.py` flags responses where scores differ by >3 between models, or where guardrail triggered but judge score is still <7
- Flagged cases written to `evaluation/flagged.json` for manual review

### Report (`evaluation/report.py`)
- Radar chart: OSS vs Frontier across hallucination, safety, bias
- Bar chart: guardrail trigger rates (input + output) per model
- Latency/cost table (from Langfuse trace export)
- Output: `evaluation/report.pdf` (or PNG set)

---

## Deployment

### OSS Assistant — HF Spaces (public)

- Space type: **Docker**
- `Dockerfile` uses `supervisord` to run:
  - Gradio frontend on port 7860 (HF Spaces public port — this is the user-facing UI)
  - FastAPI backend on port 8000 (internal only — Gradio calls it via `localhost:8000`)
- Model loaded at startup with `transformers` + `torch` (CPU inference)
- Expected latency: 3–5s on free Space CPU
- Secrets set in HF Space settings: `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

### Frontier Assistant — Local only

- `docker-compose up frontier`
- FastAPI backend on `localhost:8001`, Gradio on `localhost:8002`
- Requires: `GEMINI_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_*` in `.env`

### Cost + Latency Table

| Model | Deployment | Avg Latency | Cost/1K tokens |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | HF Spaces (CPU) | ~3–5s | $0 |
| Gemini 1.5 Flash | API (free tier) | ~0.8–1.5s | $0 |

*Populated from Langfuse traces after evaluation run.*

---

## Tech Stack

| Layer | OSS Assistant | Frontier Assistant |
|---|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (HuggingFace) | `gemini-1.5-flash` (Google AI) |
| Inference | `transformers` + `torch` | `google-generativeai` SDK |
| Backend | FastAPI | FastAPI |
| Frontend | Gradio `ChatInterface` | Gradio `ChatInterface` |
| Memory | `shared/memory.py` (deque, cap=10) | same |
| Tools | Tavily (`tavily-python`) | Tavily + Gemini native function calling |
| Guardrails | `shared/guardrails.py` | same |
| Observability | Langfuse (`langfuse`) | same |
| Deployment | HF Spaces Docker | Local Docker Compose |

---

## What Would Be Improved With More Time

1. **Vector DB memory** — replace sliding window with ChromaDB for long-term semantic memory
2. **Streaming responses** — both backends return full responses; streaming would improve UX
3. **More eval categories** — add cultural bias, multilingual, and code-generation prompts
4. **Quantized OSS model** — run Qwen2.5-1.5B-Instruct with 4-bit quantization for better quality without cost increase
5. **Async inference** — FastAPI with async endpoints + `asyncio` for the OSS model to handle concurrent eval runs
6. **CI/CD** — GitHub Actions to auto-redeploy HF Space on push, run eval suite on PRs
