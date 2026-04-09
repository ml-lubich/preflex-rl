from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from preflex_rl.dqn import DQNAgent, TrainConfig, epsilon_at_step
from preflex_rl.preferences import PreferenceWeights
from preflex_rl.shaping import ShapedCartPoleWrapper


def default_train_config() -> TrainConfig:
    return TrainConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        warmup=500,
        train_every=4,
        target_update_every=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=8_000,
        hidden_dim=128,
        max_grad_norm=10.0,
    )


def smoke_train_config() -> TrainConfig:
    """Tiny settings for CI / unit smoke tests."""
    return TrainConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=32,
        buffer_size=2_000,
        warmup=40,
        train_every=1,
        target_update_every=10,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_steps=200,
        hidden_dim=32,
        max_grad_norm=10.0,
    )


def run_training(
    *,
    total_env_steps: int,
    seed: int,
    prefs: PreferenceWeights,
    cfg: TrainConfig,
    metrics_path: Path | None,
) -> dict[str, float]:
    """Train DQN on shaped CartPole; returns scalar metrics for logging / Crew debrief."""

    rng_py = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    base = gym.make("CartPole-v1")
    shaped = ShapedCartPoleWrapper(base, prefs)
    env = gym.wrappers.RecordEpisodeStatistics(shaped)
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    obs_dim = int(np.prod(obs.shape))
    n_actions = int(env.action_space.n)

    device = torch.device("cpu")
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        device=device,
        cfg=cfg,
        rng=rng_py,
    )

    episode_returns: list[float] = []
    ep_return = 0.0
    losses: list[float] = []

    for _ in range(total_env_steps):
        eps = epsilon_at_step(agent.global_step, cfg)
        action = agent.act(obs, eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_obs = np.asarray(next_obs, dtype=np.float32)
        ep_return += float(reward)
        agent.remember(obs, action, float(reward), next_obs, done)
        agent.tick()
        loss = agent.learn_step()
        if loss is not None:
            losses.append(loss)
        obs = next_obs
        if done:
            ep_stat = float(info.get("episode", {}).get("r", ep_return))
            episode_returns.append(ep_stat)
            ep_return = 0.0
            obs, _ = env.reset(seed=int(np_rng.integers(0, 2**31 - 1)))
            obs = np.asarray(obs, dtype=np.float32)

    env.close()

    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    last_return = float(episode_returns[-1]) if episode_returns else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0

    metrics: dict[str, float] = {
        "total_env_steps": float(total_env_steps),
        "episodes_completed": float(len(episode_returns)),
        "mean_episode_return": mean_return,
        "last_episode_return": last_return,
        "mean_td_loss": mean_loss,
        "seed": float(seed),
        "velocity_l2_penalty": float(prefs.velocity_l2_penalty),
        "action_switch_penalty": float(prefs.action_switch_penalty),
    }

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflex RL — shaped DQN on CartPole")
    parser.add_argument("--steps", type=int, default=15_000, help="Environment steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefs",
        type=Path,
        default=Path("configs/preferences.yaml"),
        help="YAML with shaping weights",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("runs/metrics.json"),
        help="Where to write metrics JSON for Crew debrief / dashboards",
    )
    parser.add_argument("--smoke", action="store_true", help="Use tiny config for debugging")
    args = parser.parse_args()

    prefs = PreferenceWeights.from_yaml(args.prefs)
    cfg = smoke_train_config() if args.smoke else default_train_config()
    metrics = run_training(
        total_env_steps=args.steps,
        seed=args.seed,
        prefs=prefs,
        cfg=cfg,
        metrics_path=args.metrics_out,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.metrics_out}")


if __name__ == "__main__":
    main()
