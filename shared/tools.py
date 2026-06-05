import os
from tavily import TavilyClient

_QUESTION_WORDS = {"what", "who", "when", "where", "which", "how", "why"}
_client: TavilyClient | None = None

def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client

def needs_search(message: str) -> bool:
    first_word = message.strip().split()[0].lower().rstrip("?") if message.strip() else ""
    return first_word in _QUESTION_WORDS

def web_search(query: str) -> str:
    results = _get_client().search(query=query, max_results=3)
    snippets = [r.get("content", "") for r in results.get("results", [])]
    return "\n".join(s for s in snippets if s)
