"""CLI: metrics debrief via MiniMax (preferred), optional CrewAI, or local fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_env_files() -> None:
    from dotenv import load_dotenv

    load_dotenv(_repo_root() / ".env")
    load_dotenv()


def format_fallback_summary(metrics: dict[str, float]) -> str:
    mean_r = metrics.get("mean_episode_return", 0.0)
    last_r = metrics.get("last_episode_return", 0.0)
    loss = metrics.get("mean_td_loss", 0.0)
    vpen = metrics.get("velocity_l2_penalty", 0.0)
    apen = metrics.get("action_switch_penalty", 0.0)
    return (
        f"Training summary: mean return {mean_r:.2f}, last return {last_r:.2f}, "
        f"mean TD loss {loss:.4f}. "
        f"Shaping: velocity_l2_penalty={vpen}, action_switch_penalty={apen}. "
        "Raise penalties if the pole oscillates or the cart thrashes; "
        "lower them if returns collapse."
    )


def _crew_summary(metrics: dict[str, float]) -> str:
    from crewai import Agent, Crew, Process, Task

    payload = json.dumps(metrics, indent=2)
    analyst = Agent(
        role="RL metrics analyst",
        goal="Interpret scalar RL metrics and relate them to CartPole behavior.",
        backstory="You specialize in tabular logs from small DQN runs with reward shaping.",
        verbose=False,
    )
    writer = Agent(
        role="Technical writer",
        goal="Deliver a concise debrief for the engineer.",
        backstory="You turn numbers into actionable next steps.",
        verbose=False,
    )
    t1 = Task(
        description=f"Analyze these metrics:\n{payload}\n"
        "Call out whether returns are improving and how penalties may interact.",
        expected_output="Three bullet observations.",
        agent=analyst,
    )
    t2 = Task(
        description=(
            "Turn the observations into a four-sentence debrief with one tuning suggestion."
        ),
        expected_output="Plain text, no markdown headings.",
        agent=writer,
        context=[t1],
    )
    crew = Crew(
        agents=[analyst, writer],
        tasks=[t1, t2],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debrief Preflex RL metrics (MiniMax / CrewAI / local)",
    )
    parser.add_argument(
        "metrics_path",
        nargs="?",
        type=Path,
        default=Path("runs/metrics.json"),
    )
    args = parser.parse_args()

    load_env_files()

    path = args.metrics_path
    if not path.is_file():
        print(f"Missing metrics file: {path} (run preflex-train first)", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("metrics.json must be a JSON object", file=sys.stderr)
        sys.exit(1)
    metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    use_minimax = os.environ.get("PREFLEX_USE_MINIMAX", "1") == "1"
    if use_minimax and os.environ.get("MINIMAX_API_KEY", "").strip():
        try:
            from preflex_rl.minimax_debrief import debrief_metrics

            print(debrief_metrics(metrics))
            return
        except ImportError as exc:
            print(f"MiniMax debrief import failed: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"MiniMax debrief failed ({exc!r}); trying fallbacks.", file=sys.stderr)

    want_crew = os.environ.get("PREFLEX_USE_CREW", "0") == "1"
    if want_crew:
        try:
            print(_crew_summary(metrics))
            return
        except ImportError:
            print("Install CrewAI: pip install -e '.[crew]'", file=sys.stderr)
        except Exception as exc:
            print(f"Crew failed ({exc!r}); fallback summary:", file=sys.stderr)

    print(format_fallback_summary(metrics))


if __name__ == "__main__":
    main()
