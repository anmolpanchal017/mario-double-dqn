from __future__ import annotations

from collections import deque
import random
from typing import Any

import numpy as np

from .config import EnvConfig


class SkipFrame:
    def __init__(self, env: Any, skip: int) -> None:
        self.env = env
        self.skip = skip
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return _first_obs(self.env.reset(**kwargs))

    def step(self, action):
        total_reward = 0.0
        done = False
        trunc = False
        info = {}
        obs = None

        for _ in range(self.skip):
            result = self.env.step(action)
            obs, reward, done, trunc, info = _normalize_step(result)
            total_reward += reward
            if done or trunc:
                break
        return obs, total_reward, done, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


class ResizeObservation:
    def __init__(self, env: Any, size: int = 84, grayscale: bool = True) -> None:
        self.env = env
        self.size = size
        self.grayscale = grayscale
        self.action_space = env.action_space
        channels = 1 if grayscale else 3
        self.observation_space = (channels, size, size)

    def reset(self, **kwargs):
        return self._transform(_first_obs(self.env.reset(**kwargs)))

    def step(self, action):
        obs, reward, done, trunc, info = _normalize_step(self.env.step(action))
        return self._transform(obs), reward, done, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def _transform(self, obs):
        from PIL import Image

        image = Image.fromarray(obs)
        if self.grayscale:
            image = image.convert("L")
        image = image.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)
        if self.grayscale:
            return arr[np.newaxis, :, :]
        return np.moveaxis(arr, -1, 0)


class FrameStack:
    def __init__(self, env: Any, num_stack: int) -> None:
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.action_space = env.action_space
        channels, height, width = env.observation_space
        self.observation_space = (channels * num_stack, height, width)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, reward, done, trunc, info = _normalize_step(self.env.step(action))
        self.frames.append(obs)
        return self._get_observation(), reward, done, trunc, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)


def make_env(config: EnvConfig):
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:
        raise ImportError(
            "Mario environment dependencies are missing. Use Python 3.10/3.11 and run "
            '`python -m pip install -e ".[dev]"`.'
        ) from exc

    movements = {
        "right": RIGHT_ONLY,
        "simple": SIMPLE_MOVEMENT,
        "complex": COMPLEX_MOVEMENT,
    }
    if config.movement not in movements:
        raise ValueError(f"Unknown movement set: {config.movement}")

    env = gym_super_mario_bros.make(config.world)
    env = JoypadSpace(env, movements[config.movement])
    env = SkipFrame(env, config.skip_frames)
    env = ResizeObservation(env, config.resize, config.grayscale)
    env = FrameStack(env, config.frame_stack)
    return env


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _first_obs(reset_result):
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _normalize_step(result):
    if len(result) == 5:
        return result
    obs, reward, done, info = result
    return obs, reward, done, False, info

