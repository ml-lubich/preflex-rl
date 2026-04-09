from __future__ import annotations

import json
import os


def _strip_thinking_blocks(text: str) -> str:
    """Remove MiniMax auxiliary blocks (e.g. redacted_thinking) from the reply body."""

    out = text
    while True:
        lower = out.lower()
        start = lower.find("<redacted_thinking>")
        if start == -1:
            break
        end = lower.find("</redacted_thinking>", start)
        if end == -1:
            break
        end += len("</redacted_thinking>")
        out = out[:start] + out[end:]
    return out.strip()


def debrief_metrics(metrics: dict[str, float]) -> str:
    """Generate a short debrief using MiniMax (OpenAI-compatible API).

    Environment:
      MINIMAX_API_KEY — required
      MINIMAX_BASE_URL — default https://api.minimax.io/v1
      MINIMAX_MODEL — default MiniMax-M2
    """

    from openai import OpenAI

    api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is not set")

    base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1").strip()
    model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2").strip()
    temperature = float(os.environ.get("MINIMAX_TEMPERATURE", "1.0"))

    client = OpenAI(api_key=api_key, base_url=base_url)
    payload = json.dumps(metrics, indent=2)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an RL engineer. The user trained a DQN on CartPole with "
                    "preference-based reward shaping (YAML weights). Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize these metrics in 4–6 sentences. "
                    "Say whether learning looks healthy, mention shaping penalties if relevant, "
                    "and give one concrete next step.\n\n"
                    f"{payload}"
                ),
            },
        ],
        temperature=temperature,
    )
    choice = response.choices[0].message
    text = choice.content
    if text is None:
        return ""
    return _strip_thinking_blocks(text)
