from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

from .config import AgentConfig, resolve_device
from .model import MarioNet, torch
from .replay import ReplayBuffer


class MarioAgent:
    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        config: AgentConfig,
        save_dir: Path,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(resolve_device(config.device))
        self.online_net = MarioNet(state_dim, action_dim).to(self.device)
        self.target_net = MarioNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(config.memory_size)
        self.curr_step = 0
        self.exploration_rate = config.exploration_rate

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.exploration_rate:
            action = random.randrange(self.action_dim)
        else:
            state_t = self._state_tensor(state).unsqueeze(0)
            with torch.no_grad():
                action = int(self.online_net(state_t).argmax(dim=1).item())

        if explore:
            self.exploration_rate *= self.config.exploration_rate_decay
            self.exploration_rate = max(self.config.exploration_rate_min, self.exploration_rate)
            self.curr_step += 1
        return action

    def cache(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        self.memory.push(state, next_state, action, reward, done)

    def learn(self) -> tuple[float | None, float | None]:
        if self.curr_step < self.config.burnin:
            return None, None
        if self.curr_step % self.config.learn_every != 0:
            return None, None
        if len(self.memory) < self.config.batch_size:
            return None, None

        batch = self.memory.sample(self.config.batch_size)
        state = torch.stack([self._state_tensor(t.state) for t in batch])
        next_state = torch.stack([self._state_tensor(t.next_state) for t in batch])
        action = torch.tensor([t.action for t in batch], device=self.device).long()
        reward = torch.tensor([t.reward for t in batch], device=self.device).float()
        done = torch.tensor([t.done for t in batch], device=self.device).float()

        current_q = self.online_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next_action = self.online_net(next_state).argmax(dim=1)
            next_q = self.target_net(next_state).gather(1, best_next_action.unsqueeze(1)).squeeze(1)
            target_q = reward + (1 - done) * self.config.gamma * next_q

        loss = torch.nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.curr_step % self.config.sync_every == 0:
            self.sync_target()
        if self.curr_step % self.config.save_every == 0:
            self.save()

        return float(current_q.mean().item()), float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, name: str = "latest.pt") -> Path:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "curr_step": self.curr_step,
                "exploration_rate": self.exploration_rate,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )
        return path

    def load(self, checkpoint: str | Path) -> None:
        payload: dict[str, Any] = torch.load(checkpoint, map_location=self.device)
        self.online_net.load_state_dict(payload["online_net"])
        self.target_net.load_state_dict(payload.get("target_net", payload["online_net"]))
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.curr_step = int(payload.get("curr_step", 0))
        self.exploration_rate = float(payload.get("exploration_rate", self.config.exploration_rate_min))

    def _state_tensor(self, state: np.ndarray):
        array = np.asarray(state)
        if array.ndim == 3 and array.shape[-1] == self.state_dim[0]:
            array = np.moveaxis(array, -1, 0)
        return torch.as_tensor(array, device=self.device)
