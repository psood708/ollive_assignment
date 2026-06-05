from unittest.mock import MagicMock, patch


def test_score_returns_three_dimensions():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"hallucination": 8, "safety": 9, "bias": 9, "reasoning": "Accurate and safe."}'
    mock_model.generate_content.return_value = mock_response

    with patch("evaluation.judge._get_model", return_value=mock_model):
        from evaluation.judge import score
        result = score("What is 2+2?", "2+2 equals 4.")

    assert result["hallucination"] == 8
    assert result["safety"] == 9
    assert result["bias"] == 9
    assert "reasoning" in result


def test_score_prompt_contains_response():
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"hallucination": 5, "safety": 5, "bias": 5, "reasoning": "Mediocre."}'
    mock_model.generate_content.return_value = mock_response

    with patch("evaluation.judge._get_model", return_value=mock_model):
        from evaluation.judge import score
        score("Tell me a fact", "The sky is green.")

    call_args = mock_model.generate_content.call_args[0][0]
    assert "The sky is green." in call_args
    assert "Tell me a fact" in call_args
