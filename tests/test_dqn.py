import random

import numpy as np
import torch

from preflex_rl.dqn import DQNAgent, TrainConfig


def test_agent_act_and_learn() -> None:
    cfg = TrainConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=8,
        buffer_size=200,
        warmup=10,
        train_every=1,
        target_update_every=5,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=100,
        hidden_dim=16,
        max_grad_norm=5.0,
    )
    rng = random.Random(0)
    agent = DQNAgent(
        obs_dim=4,
        n_actions=2,
        device=torch.device("cpu"),
        cfg=cfg,
        rng=rng,
    )
    obs = np.zeros(4, dtype=np.float32)
    a = agent.act(obs, epsilon=1.0)
    assert a in (0, 1)
    obs2 = np.ones(4, dtype=np.float32)
    for _ in range(20):
        agent.remember(obs, a, 1.0, obs2, False)
    agent.tick()
    loss = agent.learn_step()
    assert loss is not None
    assert loss >= 0.0
