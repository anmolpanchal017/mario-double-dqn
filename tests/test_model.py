import pytest


torch = pytest.importorskip("torch")

from mario_dqn.model import MarioNet


def test_model_forward_shape():
    model = MarioNet((4, 84, 84), action_dim=7)
    batch = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)

    output = model(batch)

    assert output.shape == (2, 7)

