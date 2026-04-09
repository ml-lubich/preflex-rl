from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class PreferenceWeights:
    """Human-readable shaping terms (interpretable, no black-box LLM in the loop)."""

    velocity_l2_penalty: float
    action_switch_penalty: float

    @staticmethod
    def from_yaml(path: Path) -> PreferenceWeights:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("preferences YAML must be a mapping at the top level")
        v = float(raw.get("velocity_l2_penalty", 0.0))
        s = float(raw.get("action_switch_penalty", 0.0))
        return PreferenceWeights(velocity_l2_penalty=v, action_switch_penalty=s)
