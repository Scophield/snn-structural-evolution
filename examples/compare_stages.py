"""Quickly compare all structural evolution stages on MNIST."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snn_structural_evolution.data import get_mnist_loaders
from snn_structural_evolution.metrics import spike_statistics
from snn_structural_evolution.stages import get_model
from snn_structural_evolution.training import evaluate, resolve_device, seed_all, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Stage 0-4 on a short MNIST run.")
    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=40)
    parser.add_argument("--max-test-batches", type=int, default=40)
    parser.add_argument("--output-csv", type=Path, default=Path("runs/compare_stages.csv"))
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--no-download", dest="download", action="store_false", default=True)
    return parser.parse_args()


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{100 * float(value):.2f}%"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_loader, test_loader, _, _ = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        download=args.download,
        pin_memory=device.type == "cuda",
    )

    rows = []
    for stage in args.stages:
        seed_all(args.seed)
        model = get_model(stage=stage, hidden_dim=args.hidden_dim, time_steps=args.time_steps).to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_metrics = {"loss": 0.0, "accuracy": 0.0}
        for _ in range(args.epochs):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                max_batches=args.max_train_batches,
                show_progress=False,
            )
        test_metrics = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            max_batches=args.max_test_batches,
        )
        spikes = spike_statistics(model, test_loader, device, max_batches=args.max_test_batches)
        rows.append(
            {
                "stage": stage,
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "spike_rate": spikes["spike_rate"],
                "activation_sparsity": spikes["activation_sparsity"],
                "event_ops_proxy": spikes["event_ops_proxy"],
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("| Stage | Train acc | Test acc | Spike rate | Sparsity | Event ops proxy |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        event_ops = row["event_ops_proxy"] if row["event_ops_proxy"] is not None else "-"
        print(
            f"| {row['stage']} | {format_percent(row['train_accuracy'])} | "
            f"{format_percent(row['test_accuracy'])} | {format_percent(row['spike_rate'])} | "
            f"{format_percent(row['activation_sparsity'])} | {event_ops} |"
        )
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
