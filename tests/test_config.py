from pathlib import Path

from mario_dqn.config import load_config


def test_load_smoke_config():
    config = load_config(Path("configs/smoke.toml"))

    assert config.run.name == "smoke"
    assert config.run.episodes == 2
    assert config.env.frame_stack == 4
    assert config.agent.batch_size == 16

