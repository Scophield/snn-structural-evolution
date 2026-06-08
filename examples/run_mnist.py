"""Minimal example: instantiate Stage 4 and run one MNIST batch."""

from __future__ import annotations

from snn_structural_evolution.data import get_mnist_loaders
from snn_structural_evolution.stages import get_model
from snn_structural_evolution.training import resolve_device


def main() -> None:
    device = resolve_device("auto")
    model = get_model(stage=4, time_steps=4).to(device)
    _, test_loader, _, _ = get_mnist_loaders(batch_size=8, data_dir="./data")

    images, _ = next(iter(test_loader))
    logits = model(images.to(device))
    print(f"device: {device}")
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
