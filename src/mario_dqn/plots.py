from __future__ import annotations

import csv
from pathlib import Path


def generate_plots(run_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate plots.") from exc

    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_path}")

    rows = _read_metrics(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_path}")

    outputs: list[Path] = []
    series = [
        ("episode_reward", "Episode Reward", "reward.png"),
        ("mean_reward_100", "Mean Reward (100 episodes)", "mean_reward_100.png"),
        ("mean_loss_100", "Mean Loss (100 episodes)", "mean_loss_100.png"),
        ("mean_q_100", "Mean Q Value (100 episodes)", "mean_q_100.png"),
    ]
    episodes = [int(row["episode"]) for row in rows]

    for column, title, filename in series:
        values = [float(row[column]) for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(episodes, values)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)
        output = run_path / filename
        fig.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)

    return outputs


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

