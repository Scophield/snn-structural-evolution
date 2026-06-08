"""Export spike-rate statistics from a trained temporal SNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snn_structural_evolution.data import get_mnist_loaders
from snn_structural_evolution.metrics import accuracy, spike_statistics
from snn_structural_evolution.stages import get_model
from snn_structural_evolution.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export spike statistics for a temporal SNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/stage4_best.pt"))
    parser.add_argument("--stage", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=Path("runs/stage4_spike_statistics.json"))
    parser.add_argument("--no-download", dest="download", action="store_false", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = get_model(
        stage=args.stage,
        hidden_dim=args.hidden_dim,
        time_steps=args.time_steps,
        threshold=args.threshold,
    ).to(device)

    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Warning: {args.checkpoint} not found; exporting statistics from an untrained model.")

    _, test_loader, _, _ = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        pin_memory=device.type == "cuda",
    )
    stats = spike_statistics(model, test_loader, device, max_batches=args.max_batches)
    stats["accuracy"] = accuracy(model, test_loader, device, max_batches=args.max_batches)
    stats["stage"] = args.stage
    stats["time_steps"] = args.time_steps
    stats["evaluated_batches"] = args.max_batches

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
