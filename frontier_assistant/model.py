import os
from typing import Dict, List, Tuple
import google.generativeai as genai

MODEL_ID = "gemini-1.5-flash"
_model = None

def _load() -> None:
    global _model
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel(
            model_name=MODEL_ID,
            system_instruction="You are a helpful personal assistant.",
        )

def _to_gemini_history(history: List[Dict[str, str]]) -> List[Dict]:
    result = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        result.append({"role": role, "parts": [msg["content"]]})
    return result

def generate(message: str, history: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    _load()
    chat = _model.start_chat(history=_to_gemini_history(history))
    response = chat.send_message(message)
    usage = response.usage_metadata
    tokens = {
        "input": usage.prompt_token_count if usage else 0,
        "output": usage.candidates_token_count if usage else 0,
    }
    return response.text, tokens
