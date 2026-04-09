from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


Transition = tuple[npt.NDArray[np.float32], int, float, npt.NDArray[np.float32], bool]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        s: npt.NDArray[np.float32],
        a: int,
        r: float,
        s2: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(
        self, batch_size: int, rng: random.Random
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = rng.sample(self._buf, batch_size)
        s = np.stack([b[0] for b in batch], axis=0)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.stack([b[3] for b in batch], axis=0)
        d = np.array([b[4] for b in batch], dtype=np.float32)
        return (
            torch.from_numpy(s),
            torch.from_numpy(a),
            torch.from_numpy(r),
            torch.from_numpy(s2),
            torch.from_numpy(d),
        )

    def __len__(self) -> int:
        return len(self._buf)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    gamma: float
    lr: float
    batch_size: int
    buffer_size: int
    warmup: int
    train_every: int
    target_update_every: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    hidden_dim: int
    max_grad_norm: float


def epsilon_at_step(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.epsilon_decay_steps:
        return cfg.epsilon_end
    t = step / max(cfg.epsilon_decay_steps, 1)
    return cfg.epsilon_start + t * (cfg.epsilon_end - cfg.epsilon_start)


class DQNAgent:
    def __init__(
        self,
        *,
        obs_dim: int,
        n_actions: int,
        device: torch.device,
        cfg: TrainConfig,
        rng: random.Random,
    ) -> None:
        self._device = device
        self._cfg = cfg
        self._rng = rng
        self._q = QNetwork(obs_dim, n_actions, cfg.hidden_dim).to(device)
        self._target = QNetwork(obs_dim, n_actions, cfg.hidden_dim).to(device)
        self._target.load_state_dict(self._q.state_dict())
        self._opt = optim.Adam(self._q.parameters(), lr=cfg.lr)
        self._buffer = ReplayBuffer(cfg.buffer_size)
        self._global_step = 0
        self._learn_steps = 0
        self._n_actions = n_actions

    @property
    def global_step(self) -> int:
        return self._global_step

    def act(self, obs: npt.NDArray[np.float32], epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return self._rng.randint(0, self._n_actions - 1)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
            qv = self._q(t)
            return int(torch.argmax(qv, dim=1).item())

    def remember(
        self,
        s: npt.NDArray[np.float32],
        a: int,
        r: float,
        s2: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        self._buffer.push(s, a, r, s2, done)

    def learn_step(self) -> float | None:
        cfg = self._cfg
        min_buf = max(cfg.warmup, cfg.batch_size)
        if len(self._buffer) < min_buf:
            return None
        if self._global_step % cfg.train_every != 0:
            return None
        s, a, r, s2, d = self._buffer.sample(cfg.batch_size, self._rng)
        s = s.to(self._device)
        a = a.to(self._device)
        r = r.to(self._device)
        s2 = s2.to(self._device)
        d = d.to(self._device)

        q_all = self._q(s)
        q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._target(s2).max(dim=1).values
            target = r + cfg.gamma * (1.0 - d) * q_next
        loss = torch.nn.functional.mse_loss(q_sa, target)
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q.parameters(), cfg.max_grad_norm)
        self._opt.step()
        self._learn_steps += 1
        if self._learn_steps % cfg.target_update_every == 0:
            self._target.load_state_dict(self._q.state_dict())
        return float(loss.item())

    def tick(self) -> None:
        self._global_step += 1
