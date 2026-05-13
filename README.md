# Preflex RL

**Preflex** = *preference-flexible* reinforcement learning: a **real DQN** trainer on **CartPole-v1** where you steer behavior with **interpretable YAML preferences** (velocity smoothing, anti-thrashing), not a template-only repo.

```mermaid
flowchart LR
    YAML[("🎚 configs/preferences.yaml")]
    CLI{{"💻 preflex-train · preflex-debrief"}}
    SHAPE["⚖️ shaping.py<br/>CartPole reward shaping"]
    DQN["🧠 dqn.py<br/>replay + target net + ε-greedy"]
    METR[/"📊 runs/metrics.json"/]
    DEB["📝 debrief.py"]
    MM["🛰 minimax_debrief.py"]
    LLM(("🤖 MiniMax / Crew"))
    REPORT[/"📄 debrief output"/]

    YAML --> SHAPE
    CLI --> SHAPE --> DQN --> METR
    METR --> DEB --> MM --> LLM --> REPORT
    DEB -. local fallback .-> REPORT

    classDef io fill:#0e1116,stroke:#2f81f7,stroke-width:1.5px,color:#e6edf3;
    classDef brain fill:#161b22,stroke:#d29922,stroke-width:1.5px,color:#e6edf3;
    classDef tool fill:#161b22,stroke:#3fb950,stroke-width:1.5px,color:#e6edf3;
    classDef out fill:#0e1116,stroke:#a371f7,stroke-width:1.5px,color:#e6edf3;
    class CLI brain;
    class YAML,LLM io;
    class SHAPE,DQN,DEB,MM tool;
    class METR,REPORT out;
```

## Table of contents

- [What this is](#what-this-is)
- [Training loop (algorithm)](#training-loop-algorithm)
- [Debrief sequence](#debrief-sequence)
- [Quick start](#quick-start)
- [Layout](#layout)
- [License](#license)
- [🗺️ Repository map](#️-repository-map)
- [📊 Code composition](#-code-composition)

## Training loop (algorithm)

```mermaid
flowchart LR
    A([preflex-train])
    B["load preferences.yaml<br/>shaping weights"]
    C["build CartPole + ShapingWrapper"]
    D["reset DQN<br/>policy + target net + replay"]
    E["ε-greedy action"]
    F["step env<br/>shape reward"]
    G["push to replay buffer"]
    H{"buffer warm?"}
    I["sample batch<br/>compute TD loss"]
    J["backprop policy net"]
    K{"sync interval?"}
    L["copy policy → target"]
    M{"steps left?"}
    N["write metrics.json"]
    Z([done])
    A --> B --> C --> D --> E --> F --> G --> H
    H -- no  --> M
    H -- yes --> I --> J --> K
    K -- yes --> L --> M
    K -- no  --> M
    M -- yes --> E
    M -- no  --> N --> Z
```

## Debrief sequence

```mermaid
sequenceDiagram
    participant U as user
    participant D as preflex-debrief
    participant FS as runs/metrics.json
    participant MM as MiniMax
    participant CR as CrewAI
    participant LOC as local fallback

    U->>D: preflex-debrief metrics.json
    D->>FS: read metrics
    alt MINIMAX_API_KEY + PREFLEX_USE_MINIMAX!=0
        D->>MM: chat(metrics summary)
        MM-->>D: narrative
    else PREFLEX_USE_CREW=1
        D->>CR: crew.kickoff()
        CR-->>D: narrative
    else
        D->>LOC: deterministic summary
        LOC-->>D: narrative
    end
    D-->>U: debrief output
```

## What this is

- **RL core:** PyTorch DQN with replay buffer, target network, ε-greedy exploration.
- **Idea:** Add **reward shaping** from human-readable weights (`configs/preferences.yaml`) so you can trade off pole stability vs raw return before touching network code.
- **MiniMax debrief (default):** After training, `preflex-debrief` calls the **MiniMax OpenAI-compatible API** using `MINIMAX_*` variables from `.env` (see `.env.example`). Set `PREFLEX_USE_MINIMAX=0` to skip the API and fall back to a local summary.
- **CrewAI (optional):** Install `pip install -e '.[crew]'` and set `PREFLEX_USE_CREW=1` if you prefer a Crew crew instead; MiniMax is tried first when `MINIMAX_API_KEY` is set.

### Secrets

- Copy `.env.example` → `.env` and add your keys. **`.env` is gitignored** — never commit it.
- If a key was pasted into chat or a ticket, **rotate it** in the MiniMax console and update `.env`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
cp .env.example .env   # then edit with MINIMAX_API_KEY
preflex-train --steps 15000 --metrics-out runs/metrics.json
preflex-debrief runs/metrics.json
```

Same as: `python -m preflex_rl` (runs the debrief CLI).

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
| `src/preflex_rl/minimax_debrief.py` | MiniMax chat debrief |
| `src/preflex_rl/debrief.py` | CLI: MiniMax → CrewAI → local fallback |
| `configs/preferences.yaml` | Tunable preference weights |
| `scripts/crew_debrief.py` | Thin wrapper → `preflex_rl.debrief` |

## License

MIT


## 🗺️ Repository map

Top-level layout of `preflex-rl` rendered as a Mermaid mindmap (auto-generated from the on-disk tree).

```mermaid
mindmap
  root((preflex-rl))
    configs/
      preferences.yaml
    scripts/
      crew_debrief.py
    src/
      preflex_rl
    tests/
      test_debrief.py
      test_dqn.py
      test_minimax_debrief.py
      test_minimax_strip.py
      test_preferences.py
      test_shaping.py
    files
      LICENSE
      README.md
      pyproject.toml
```


## 📊 Code composition

File-type breakdown of source under this repo (skips `.git`, `node_modules`, build caches, lockfiles).

```mermaid
pie showData title File-type composition of preflex-rl (20 files)
    "Python" : 16
    "Other" : 1
    "TOML" : 1
    "Markdown" : 1
    "YAML" : 1
```
