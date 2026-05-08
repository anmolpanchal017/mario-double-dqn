import pytest


torch = pytest.importorskip("torch")

from mario_dqn.agent import MarioAgent
from mario_dqn.config import AgentConfig


def test_agent_random_action_respects_action_space(tmp_path):
    config = AgentConfig(
        batch_size=2,
        memory_size=10,
        gamma=0.9,
        learning_rate=0.00025,
        burnin=100,
        learn_every=4,
        sync_every=100,
        save_every=1000,
        exploration_rate=1.0,
        exploration_rate_decay=1.0,
        exploration_rate_min=1.0,
        device="cpu",
    )
    agent = MarioAgent((4, 84, 84), action_dim=3, config=config, save_dir=tmp_path)
    action = agent.act(torch.zeros((4, 84, 84), dtype=torch.uint8).numpy())

    assert action in {0, 1, 2}

