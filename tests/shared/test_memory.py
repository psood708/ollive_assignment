from shared.memory import ConversationMemory

def test_add_and_get_single_turn():
    mem = ConversationMemory(max_messages=20)
    mem.add("s1", "user", "hello")
    mem.add("s1", "assistant", "hi there")
    history = mem.get("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}

def test_sliding_window_drops_oldest():
    mem = ConversationMemory(max_messages=4)
    for i in range(6):
        mem.add("s1", "user", f"msg {i}")
    history = mem.get("s1")
    assert len(history) == 4
    assert history[0]["content"] == "msg 2"

def test_sessions_are_isolated():
    mem = ConversationMemory()
    mem.add("s1", "user", "hello")
    mem.add("s2", "user", "world")
    assert mem.get("s1")[0]["content"] == "hello"
    assert mem.get("s2")[0]["content"] == "world"
    assert len(mem.get("s1")) == 1

def test_empty_session_returns_empty_list():
    mem = ConversationMemory()
    assert mem.get("nonexistent") == []

def test_clear_removes_session():
    mem = ConversationMemory()
    mem.add("s1", "user", "hello")
    mem.clear("s1")
    assert mem.get("s1") == []

def test_clear_nonexistent_does_not_raise():
    mem = ConversationMemory()
    mem.clear("nonexistent")  # Should not raise
