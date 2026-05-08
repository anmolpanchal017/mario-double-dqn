from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Iterable

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    next_state: np.ndarray
    action: int
    reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self._memory: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self._memory.append(
            Transition(
                state=np.asarray(state),
                next_state=np.asarray(next_state),
                action=int(action),
                reward=float(reward),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._memory):
            raise ValueError("Cannot sample more transitions than the buffer contains.")
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)

    def __iter__(self) -> Iterable[Transition]:
        return iter(self._memory)

