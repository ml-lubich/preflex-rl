from preflex_rl.minimax_debrief import _strip_thinking_blocks


def test_strip_redacted_thinking_block() -> None:
    raw = (
        "<redacted_thinking>internal\nnotes</redacted_thinking>\n\n"
        "**Summary:** Hello from the model."
    )
    assert "internal" not in _strip_thinking_blocks(raw)
    assert "**Summary:**" in _strip_thinking_blocks(raw)
