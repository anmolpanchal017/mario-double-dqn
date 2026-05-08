import numpy as np
import pytest

from mario_dqn.replay import ReplayBuffer


def test_replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=2)
    state = np.zeros((4, 84, 84), dtype=np.uint8)

    buffer.push(state, state, 0, 1.0, False)
    buffer.push(state + 1, state, 1, 2.0, True)
    buffer.push(state + 2, state, 2, 3.0, False)

    assert len(buffer) == 2
    sample = buffer.sample(1)
    assert len(sample) == 1
    assert sample[0].action in {1, 2}


def test_replay_buffer_rejects_oversized_sample():
    buffer = ReplayBuffer(capacity=1)

    with pytest.raises(ValueError):
        buffer.sample(1)

