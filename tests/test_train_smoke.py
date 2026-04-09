from pathlib import Path

from preflex_rl.preferences import PreferenceWeights
from preflex_rl.train import run_training, smoke_train_config


def test_run_training_smoke() -> None:
    root = Path(__file__).resolve().parent.parent
    prefs = PreferenceWeights.from_yaml(root / "configs/preferences.yaml")
    metrics = run_training(
        total_env_steps=250,
        seed=1,
        prefs=prefs,
        cfg=smoke_train_config(),
        metrics_path=None,
    )
    assert metrics["total_env_steps"] == 250.0
    assert metrics["episodes_completed"] >= 1.0
