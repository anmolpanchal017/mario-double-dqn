from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .agent import MarioAgent
from .config import ProjectConfig
from .env import make_env, seed_everything


def evaluate(
    config: ProjectConfig,
    checkpoint: str | Path,
    episodes: int = 1,
    render: bool = False,
    fps: float = 0.0,
    window_scale: int = 1,
) -> list[float]:
    seed_everything(config.run.seed)
    env = make_env(config.env)
    agent = MarioAgent(env.observation_space, env.action_space.n, config.agent, Path(checkpoint).parent.parent)
    agent.load(checkpoint)
    agent.exploration_rate = 0.0
    rewards: list[float] = []
    frame_delay = 1.0 / fps if fps > 0 else 0.0

    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            while True:
                if render:
                    env.render()
                    _resize_viewer(env, window_scale)
                    if frame_delay:
                        time.sleep(frame_delay)
                action = agent.act(state, explore=False)
                next_state, reward, done, trunc, info = env.step(action)
                total_reward += float(reward)
                state = next_state
                steps += 1
                max_steps = config.run.max_steps_per_episode
                if done or trunc or info.get("flag_get", False) or (max_steps and steps >= max_steps):
                    break
            rewards.append(total_reward)
            print(f"Evaluation episode {episode}: reward={total_reward:.1f}")
    finally:
        env.close()

    return rewards


def _resize_viewer(env: Any, window_scale: int) -> None:
    if window_scale <= 1:
        return

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    viewer = getattr(base_env, "viewer", None)
    window = getattr(viewer, "_window", None)
    if window is None:
        return

    width = getattr(viewer, "width", 256) * window_scale
    height = getattr(viewer, "height", 240) * window_scale
    if window.width != width or window.height != height:
        window.set_size(width, height)
