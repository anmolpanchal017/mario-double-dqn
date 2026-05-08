from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    episodes: int
    max_steps_per_episode: int
    save_dir: Path
    checkpoint_every: int
    record_every: int
    render: bool


@dataclass(frozen=True)
class EnvConfig:
    world: str
    movement: str
    frame_stack: int
    skip_frames: int
    resize: int
    grayscale: bool


@dataclass(frozen=True)
class AgentConfig:
    batch_size: int
    memory_size: int
    gamma: float
    learning_rate: float
    burnin: int
    learn_every: int
    sync_every: int
    save_every: int
    exploration_rate: float
    exploration_rate_decay: float
    exploration_rate_min: float
    device: str


@dataclass(frozen=True)
class ProjectConfig:
    run: RunConfig
    env: EnvConfig
    agent: AgentConfig


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    run = raw.get("run", {})
    env = raw.get("env", {})
    agent = raw.get("agent", {})

    return ProjectConfig(
        run=RunConfig(
            name=str(run["name"]),
            seed=int(run["seed"]),
            episodes=int(run["episodes"]),
            max_steps_per_episode=int(run.get("max_steps_per_episode", 0)),
            save_dir=Path(run["save_dir"]),
            checkpoint_every=int(run["checkpoint_every"]),
            record_every=int(run["record_every"]),
            render=bool(run.get("render", False)),
        ),
        env=EnvConfig(
            world=str(env["world"]),
            movement=str(env["movement"]),
            frame_stack=int(env["frame_stack"]),
            skip_frames=int(env["skip_frames"]),
            resize=int(env["resize"]),
            grayscale=bool(env.get("grayscale", True)),
        ),
        agent=AgentConfig(
            batch_size=int(agent["batch_size"]),
            memory_size=int(agent["memory_size"]),
            gamma=float(agent["gamma"]),
            learning_rate=float(agent["learning_rate"]),
            burnin=int(agent["burnin"]),
            learn_every=int(agent["learn_every"]),
            sync_every=int(agent["sync_every"]),
            save_every=int(agent["save_every"]),
            exploration_rate=float(agent["exploration_rate"]),
            exploration_rate_decay=float(agent["exploration_rate_decay"]),
            exploration_rate_min=float(agent["exploration_rate_min"]),
            device=str(agent.get("device", "auto")),
        ),
    )


def resolve_device(device: str):
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"

