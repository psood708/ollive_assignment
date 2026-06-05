import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from oss_assistant.model import generate
from shared.memory import ConversationMemory
from shared import tools as _tools
from shared.tools import needs_search
from shared.guardrails import check_input, check_output
from shared.observability import log_trace

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
memory = ConversationMemory(max_messages=20)
_start_time = time.time()
_request_count = 0

app = FastAPI(title="OSS Assistant Backend")

class ChatRequest(BaseModel):
    message: str
    session_id: str
    use_search: bool = True

class ChatResponse(BaseModel):
    reply: str
    tool_used: str
    tool_query: Optional[str]
    latency_ms: int
    tokens_used: int
    guardrail_triggered: bool
    session_id: str

@app.get("/health")
def health():
    return {"model": MODEL_NAME, "uptime_s": int(time.time() - _start_time), "request_count": _request_count}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _request_count
    _request_count += 1
    t0 = time.time()

    triggered, safe_reply = check_input(req.message)
    if triggered:
        log_trace(model=MODEL_NAME, session_id=req.session_id, latency_ms=int((time.time()-t0)*1000),
                  tokens_in=0, tokens_out=0, tool_used=False, tool_query=None,
                  guardrail_input=True, guardrail_output=False)
        return ChatResponse(reply=safe_reply, tool_used="none", tool_query=None,
                            latency_ms=int((time.time()-t0)*1000), tokens_used=0,
                            guardrail_triggered=True, session_id=req.session_id)

    history = memory.get(req.session_id)
    tool_used, tool_query, user_message = "none", None, req.message

    if req.use_search and needs_search(req.message):
        tool_query = req.message
        result = _tools.web_search(tool_query)
        user_message = f"[Search result: {result}]\n\n{req.message}"
        tool_used = "web_search"

    reply, tokens = generate(user_message, history)
    out_triggered, reply = check_output(reply)

    memory.add(req.session_id, "user", req.message)
    memory.add(req.session_id, "assistant", reply)
    latency_ms = int((time.time() - t0) * 1000)

    log_trace(model=MODEL_NAME, session_id=req.session_id, latency_ms=latency_ms,
              tokens_in=tokens["input"], tokens_out=tokens["output"],
              tool_used=tool_used != "none", tool_query=tool_query,
              guardrail_input=False, guardrail_output=out_triggered)

    return ChatResponse(reply=reply, tool_used=tool_used, tool_query=tool_query,
                        latency_ms=latency_ms, tokens_used=tokens["input"]+tokens["output"],
                        guardrail_triggered=out_triggered, session_id=req.session_id)
