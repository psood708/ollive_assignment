from unittest.mock import MagicMock, patch
from shared.tools import needs_search, web_search

def test_needs_search_detects_question_words():
    assert needs_search("What is the capital of France?") is True
    assert needs_search("Who invented the telephone?") is True
    assert needs_search("When did WWII end?") is True
    assert needs_search("Where is the Eiffel Tower?") is True
    assert needs_search("How does photosynthesis work?") is True

def test_needs_search_ignores_non_questions():
    assert needs_search("Tell me a joke") is False
    assert needs_search("Hello, how are you?") is False
    assert needs_search("Write me a poem about cats") is False

def test_needs_search_case_insensitive():
    assert needs_search("WHAT is gravity?") is True

def test_web_search_returns_joined_snippets():
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {"content": "Paris is the capital of France."},
            {"content": "France is a country in Western Europe."},
            {"content": "The Eiffel Tower is in Paris."},
        ]
    }
    with patch("shared.tools._get_client", return_value=mock_client):
        result = web_search("capital of France")
    assert "Paris is the capital of France." in result
    mock_client.search.assert_called_once_with(query="capital of France", max_results=3)

def test_web_search_handles_empty_results():
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    with patch("shared.tools._get_client", return_value=mock_client):
        result = web_search("obscure query")
    assert result == ""
