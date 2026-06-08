"""Shared model blocks used by the evolution stages."""

from __future__ import annotations

import torch
from torch import nn


class TwoLayerMNISTBackbone(nn.Module):
    """Small classifier backbone used to keep all stages directly comparable."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 10,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def flatten_images(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.flatten(inputs, start_dim=1)

    def flatten_temporal_images(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.flatten(inputs, start_dim=2)
