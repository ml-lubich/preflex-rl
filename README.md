# Preflex RL

**Preflex** = *preference-flexible* reinforcement learning: a **real DQN** trainer on **CartPole-v1** where you steer behavior with **interpretable YAML preferences** (velocity smoothing, anti-thrashing), not a template-only repo.

## What this is

- **RL core:** PyTorch DQN with replay buffer, target network, ε-greedy exploration.
- **Idea:** Add **reward shaping** from human-readable weights (`configs/preferences.yaml`) so you can trade off pole stability vs raw return before touching network code.
- **AI crew (optional):** After training, `scripts/crew_debrief.py` can run a **CrewAI** two-agent debrief on `runs/metrics.json` when you install `pip install -e '.[crew]'` and set `PREFLEX_USE_CREW=1` plus your LLM keys per [CrewAI docs](https://docs.crewai.com/). Otherwise it prints a deterministic summary (no API key).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
preflex-train --steps 15000 --metrics-out runs/metrics.json
python scripts/crew_debrief.py runs/metrics.json
```

Smoke / CI-sized run:

```bash
preflex-train --smoke --steps 400 --metrics-out runs/metrics.json
```

## Layout

| Path | Role |
|------|------|
| `src/preflex_rl/dqn.py` | DQN + replay |
| `src/preflex_rl/shaping.py` | CartPole shaping wrapper |
| `src/preflex_rl/train.py` | Training loop + metrics JSON |
| `configs/preferences.yaml` | Tunable preference weights |
| `scripts/crew_debrief.py` | Optional CrewAI narrative over metrics |

## License

MIT
