import gymnasium as gym

from preflex_rl.preferences import PreferenceWeights
from preflex_rl.shaping import ShapedCartPoleWrapper


def test_shaping_reports_bonus_in_info() -> None:
    prefs = PreferenceWeights(velocity_l2_penalty=0.01, action_switch_penalty=0.0)
    base = gym.make("CartPole-v1")
    env = ShapedCartPoleWrapper(base, prefs)
    obs, _ = env.reset(seed=0)
    _obs2, _r, _term, _trunc, info = env.step(0)
    assert "preflex_shaping_bonus" in info
