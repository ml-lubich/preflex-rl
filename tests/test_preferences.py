from pathlib import Path

from preflex_rl.preferences import PreferenceWeights


def test_load_preferences(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("velocity_l2_penalty: 0.1\naction_switch_penalty: 0.2\n", encoding="utf-8")
    w = PreferenceWeights.from_yaml(p)
    assert w.velocity_l2_penalty == 0.1
    assert w.action_switch_penalty == 0.2
