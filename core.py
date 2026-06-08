"""Compatibility helpers for the structural evolution toolkit."""

from __future__ import annotations

import torch

from snn_structural_evolution.stages import get_model


def convert_to_snn(stage: int = 4, **kwargs):
    """Return one of the structural evolution stages as an SNN-style model."""
    return get_model(stage=stage, **kwargs)


def simulate(model: torch.nn.Module, inputs: torch.Tensor, device: str | torch.device = "auto"):
    """Run a forward pass with automatic device placement."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model(inputs.to(device))
