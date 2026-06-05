from unittest.mock import MagicMock, patch
import shared.observability as obs

def _call_log(**overrides):
    defaults = dict(
        model="test-model", session_id="s1", latency_ms=100,
        tokens_in=10, tokens_out=20, tool_used=False, tool_query=None,
        guardrail_input=False, guardrail_output=False,
    )
    obs.log_trace(**{**defaults, **overrides})

def test_log_trace_no_ops_without_env_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    obs._client = None
    _call_log()  # Should not raise

def test_log_trace_calls_langfuse_trace(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    mock_lf = MagicMock()
    obs._client = None
    with patch("shared.observability.Langfuse", return_value=mock_lf):
        _call_log()
    mock_lf.trace.assert_called_once()
    call_kwargs = mock_lf.trace.call_args.kwargs
    assert call_kwargs["session_id"] == "s1"
    assert call_kwargs["metadata"]["model"] == "test-model"
    assert call_kwargs["metadata"]["latency_ms"] == 100

def test_log_trace_includes_judge_scores(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    mock_lf = MagicMock()
    obs._client = None
    scores = {"hallucination": 8, "safety": 9, "bias": 9}
    with patch("shared.observability.Langfuse", return_value=mock_lf):
        _call_log(judge_scores=scores)
    meta = mock_lf.trace.call_args.kwargs["metadata"]
    assert meta["judge_scores"] == scores
