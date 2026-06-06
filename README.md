# Ollive — Dual AI Personal Assistant

Two personal assistants with identical capabilities, compared head-to-head.

| | OSS Assistant | Frontier Assistant |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | Gemini 1.5 Flash |
| Deployment | HF Spaces (public) | Local only |
| Inference cost | $0 | $0 (free tier) |
| Avg latency | ~3–5s | ~0.8–1.5s |

## Features

- Multi-turn conversations with sliding-window memory (last 20 messages)
- Web search via Tavily (togglable per message)
- Input guardrails: jailbreak/prompt-injection blocklist
- Output guardrails: PII redaction + toxic content filter
- Langfuse observability (traces, latency, token counts)
- 45-prompt evaluation battery with LLM-as-judge scoring

## Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Tavily API key](https://tavily.com) (free tier: 1000 searches/month)
- [Gemini API key](https://aistudio.google.com) (free tier)
- [Langfuse account](https://langfuse.com) (free cloud tier)

### Install

```bash
git clone <this-repo> && cd ollive
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your API keys
```

### Run locally

```bash
# Start both backends
docker-compose up

# Run OSS frontend (new terminal)
PYTHONPATH=. python oss_assistant/frontend/app.py  # → http://localhost:7860

# Run Frontier frontend (new terminal)
PYTHONPATH=. python frontier_assistant/frontend/app.py  # → http://localhost:7861
```

### Run tests

```bash
pytest tests/ -v
```

### Run evaluation (requires both backends running)

```bash
PYTHONPATH=. python evaluation/run_eval.py    # ~45 min, saves evaluation/results.json
PYTHONPATH=. python evaluation/report.py      # generates charts + prints summary
```

## Architecture Decisions

**FastAPI + Gradio** — FastAPI provides a clean HTTP API so evaluation scripts are model-agnostic; Gradio provides zero-configuration chat UI. Both run in one Docker container via supervisord — Gradio on port 7860 (public), FastAPI on port 8000 (internal).

**Shared modules** — `shared/` (memory, tools, guardrails, observability) is imported by both backends identically. Evaluation differences are attributable to the model, not infrastructure.

**Sliding window memory** — `collections.deque` capped at 20 messages (~10 turns). Simple, deterministic, fast. Both assistants use it identically.

**LLM-as-judge** — Gemini 1.5 Flash scores each (prompt, response) pair on hallucination, safety, and bias (0–10 JSON). Free, fast, and produces structured output natively.

## Tradeoffs

- **Qwen2.5-0.5B on CPU** — free HF Spaces has no GPU, so inference is 3–5s. ZeroGPU or Modal would cut this to ~0.5s but adds complexity.
- **Regex guardrails** — the input blocklist catches obvious jailbreaks but not sophisticated ones. Llama Guard 2 would be more robust.
- **Single-turn judge** — scores each response in isolation; doesn't evaluate multi-turn coherence.

## What I'd improve with more time

1. Vector DB memory (ChromaDB) for long-term semantic recall
2. Streaming responses in Gradio for better perceived latency
3. Llama Guard 2 for output safety instead of the toxic-comment classifier
4. GitHub Actions CI: auto-redeploy HF Space on push, run eval suite on PRs
5. More eval categories: multilingual, code generation, cultural bias
