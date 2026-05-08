from __future__ import annotations


def _torch_nn():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the Q-network. Install the project with "
            '`python -m pip install -e ".[dev]"` in Python 3.10 or 3.11.'
        ) from exc
    return torch, nn


torch, nn = _torch_nn()


class MarioNet(nn.Module):
    """Convolutional Q-network for stacked 84x84 Mario frames."""

    def __init__(self, input_shape: tuple[int, int, int], action_dim: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        if height != 84 or width != 84:
            raise ValueError("MarioNet expects 84x84 preprocessed frames.")

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1:
            x = x / 255.0
        return self.head(self.features(x))

