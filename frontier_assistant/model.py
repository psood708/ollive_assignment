import os
from typing import Dict, List, Tuple
from google import genai
from google.genai import types

MODEL_ID = "gemini-1.5-flash"
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _to_contents(message: str, history: List[Dict[str, str]]) -> List[types.Content]:
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    return contents


def generate(message: str, history: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=_to_contents(message, history),
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful personal assistant.",
        ),
    )
    usage = response.usage_metadata
    tokens = {
        "input": usage.prompt_token_count if usage else 0,
        "output": usage.candidates_token_count if usage else 0,
    }
    return response.text, tokens
