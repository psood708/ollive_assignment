import os
import uuid
import gradio as gr
import httpx

BACKEND_URL = os.environ.get("OSS_BACKEND_URL", "http://localhost:8000")

def chat(message: str, history: list, session_id: str, use_search: bool) -> str:
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "session_id": session_id, "use_search": use_search},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["reply"]
        if data["tool_used"] == "web_search":
            reply += f"\n\n*[Web search: {data['tool_query']}]*"
        if data["guardrail_triggered"]:
            reply += "\n\n*[Safety filter applied]*"
        return reply
    except Exception as e:
        return f"Error connecting to backend: {e}"

with gr.Blocks(title="OSS Assistant — Qwen2.5-0.5B") as demo:
    gr.Markdown("# OSS Personal Assistant\nPowered by **Qwen2.5-0.5B-Instruct** · Deployed on HF Spaces")
    session_id = gr.State(lambda: uuid.uuid4().hex)
    use_search = gr.Checkbox(value=True, label="Enable web search")
    gr.ChatInterface(
        fn=lambda msg, hist, sid, search: chat(msg, hist, sid, search),
        additional_inputs=[session_id, use_search],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
