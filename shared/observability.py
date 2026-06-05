import os
from typing import Optional
from langfuse import Langfuse

_client: Optional[Langfuse] = None

def _get_client() -> Optional[Langfuse]:
    global _client
    if _client is None:
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sk = os.environ.get("LANGFUSE_SECRET_KEY")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if pk and sk:
            _client = Langfuse(public_key=pk, secret_key=sk, host=host)
    return _client

def log_trace(
    model: str,
    session_id: str,
    latency_ms: int,
    tokens_in: int,
    tokens_out: int,
    tool_used: bool,
    tool_query: Optional[str],
    guardrail_input: bool,
    guardrail_output: bool,
    judge_scores: Optional[dict] = None,
) -> None:
    client = _get_client()
    if client is None:
        return
    metadata = {
        "model": model,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_used": tool_used,
        "tool_query": tool_query,
        "guardrail_input_triggered": guardrail_input,
        "guardrail_output_triggered": guardrail_output,
    }
    if judge_scores:
        metadata["judge_scores"] = judge_scores
    client.trace(name="chat", session_id=session_id, metadata=metadata)
    client.flush()
