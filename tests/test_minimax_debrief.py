from unittest.mock import MagicMock, patch

from preflex_rl.minimax_debrief import debrief_metrics


def test_debrief_metrics_uses_openai_client(monkeypatch: object) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Summary line."

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_cls.return_value = mock_client

        out = debrief_metrics({"mean_episode_return": 1.0})
        assert out == "Summary line."
        mock_client_cls.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
