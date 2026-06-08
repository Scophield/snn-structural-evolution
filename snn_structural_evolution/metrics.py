"""Evaluation metrics for structural SNN experiments."""

from __future__ import annotations

import torch


def accuracy(model: torch.nn.Module, data_loader, device: torch.device, max_batches: int | None = None) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    seen = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)

    return correct / max(seen, 1)


def spike_statistics(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int | None]:
    """Compute spike-rate and sparsity statistics for temporal SNN stages."""
    if not hasattr(model, "spike_sequence"):
        return {
            "spike_rate": None,
            "activation_sparsity": None,
            "total_spikes": None,
            "possible_spikes": None,
            "event_ops_proxy": None,
        }

    model.eval()
    total_spikes = 0.0
    possible_spikes = 0
    num_classes = getattr(model, "num_classes", 10)

    with torch.no_grad():
        for batch_index, (images, _) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            spikes = model.spike_sequence(images.to(device, non_blocking=True))
            total_spikes += spikes.sum().item()
            possible_spikes += spikes.numel()

    spike_rate = total_spikes / max(possible_spikes, 1)
    return {
        "spike_rate": spike_rate,
        "activation_sparsity": 1.0 - spike_rate,
        "total_spikes": int(total_spikes),
        "possible_spikes": int(possible_spikes),
        "event_ops_proxy": int(total_spikes * num_classes),
    }
