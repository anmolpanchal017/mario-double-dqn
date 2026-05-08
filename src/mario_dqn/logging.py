from __future__ import annotations

from collections import deque
import csv
import datetime as dt
from pathlib import Path
import time


class MetricLogger:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.fields = [
            "episode",
            "step",
            "epsilon",
            "episode_reward",
            "episode_length",
            "mean_reward_100",
            "mean_loss_100",
            "mean_q_100",
            "elapsed_seconds",
            "timestamp",
        ]
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_losses: list[float] = []
        self.ep_qs: list[float] = []
        self.recent_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        self.recent_qs = deque(maxlen=100)
        self.start_time = time.time()

        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writeheader()

    def log_step(self, reward: float, loss: float | None, q_value: float | None) -> None:
        self.ep_rewards.append(float(reward))
        self.ep_lengths.append(1)
        if loss is not None:
            self.ep_losses.append(float(loss))
            self.recent_losses.append(float(loss))
        if q_value is not None:
            self.ep_qs.append(float(q_value))
            self.recent_qs.append(float(q_value))

    def log_episode(self, episode: int, step: int, epsilon: float) -> dict[str, float | int | str]:
        episode_reward = sum(self.ep_rewards)
        episode_length = len(self.ep_lengths)
        self.recent_rewards.append(episode_reward)
        row = {
            "episode": episode,
            "step": step,
            "epsilon": epsilon,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "mean_reward_100": _mean(self.recent_rewards),
            "mean_loss_100": _mean(self.recent_losses),
            "mean_q_100": _mean(self.recent_qs),
            "elapsed_seconds": round(time.time() - self.start_time, 3),
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        }

        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow(row)

        self.ep_rewards.clear()
        self.ep_lengths.clear()
        self.ep_losses.clear()
        self.ep_qs.clear()
        return row


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}-{run_name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))

