from __future__ import annotations

from typing import Any, SupportsFloat, SupportsInt

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from preflex_rl.preferences import PreferenceWeights


class ShapedCartPoleWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Adds preference-based shaping on top of CartPole-v1 (continuous obs, discrete actions)."""

    def __init__(self, env: gym.Env[Any, Any], prefs: PreferenceWeights) -> None:
        super().__init__(env)
        self._prefs = prefs
        self._last_action: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._last_action = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: SupportsInt
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        a = int(action)
        bonus = self._shape_bonus(obs=np.asarray(obs, dtype=np.float32), action=a)
        shaped = float(reward) + bonus
        self._last_action = a
        info = dict(info)
        info["preflex_shaping_bonus"] = bonus
        return obs, shaped, terminated, truncated, info

    def _shape_bonus(self, obs: npt.NDArray[np.float32], action: int) -> float:
        # CartPole: x, x_dot, theta, theta_dot
        vel_sq = float(obs[1] ** 2 + obs[3] ** 2)
        bonus = -self._prefs.velocity_l2_penalty * vel_sq
        if self._last_action is not None and self._prefs.action_switch_penalty > 0.0:
            if action != self._last_action:
                bonus -= self._prefs.action_switch_penalty
        return bonus
