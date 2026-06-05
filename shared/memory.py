from collections import deque
from typing import Dict, List


class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self._sessions: Dict[str, deque] = {}
        self._max_messages = max_messages

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self._max_messages)
        self._sessions[session_id].append({"role": role, "content": content})

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
