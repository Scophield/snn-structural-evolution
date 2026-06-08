"""Dataset utilities."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir: str | Path = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool = False,
):
    """Create train and test dataloaders for MNIST."""
    transform = transforms.Compose([transforms.ToTensor()])
    root = Path(data_dir).expanduser()

    train_set = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_set = datasets.MNIST(root=root, train=False, transform=transform, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader, len(train_set), len(test_set)
