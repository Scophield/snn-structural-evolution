"""Training and evaluation helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from tqdm import tqdm


def resolve_device(device_name: str | torch.device = "auto") -> torch.device:
    """Resolve an explicit or automatic torch device."""
    if str(device_name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def seed_all(seed: int = 35) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    show_progress: bool = True,
):
    """Train for one epoch and return aggregate metrics."""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    iterator = tqdm(data_loader, desc="train", leave=False, disable=not show_progress)
    for batch_index, (images, targets) in enumerate(iterator):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        seen += batch_size
        iterator.set_postfix(loss=total_loss / max(seen, 1), acc=correct / max(seen, 1))

    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}


def evaluate(
    model: torch.nn.Module,
    data_loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
):
    """Evaluate a model and return aggregate metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            seen += batch_size

    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}
