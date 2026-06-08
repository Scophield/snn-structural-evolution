"""Command-line entrypoint for MNIST structural evolution experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from snn_structural_evolution.data import get_mnist_loaders
from snn_structural_evolution.stages import get_model
from snn_structural_evolution.training import evaluate, resolve_device, seed_all, train_one_epoch


DEFAULTS = {
    "stage": 4,
    "dataset": "mnist",
    "data_dir": "./data",
    "epochs": 100,
    "batch_size": 64,
    "lr": 0.005,
    "time_steps": 4,
    "threshold": 1.0,
    "hidden_dim": 10,
    "seed": 35,
    "num_workers": 0,
    "output_dir": "./runs",
    "metrics_csv": None,
    "device": "auto",
    "download": True,
    "dry_run": False,
    "max_train_batches": None,
    "max_test_batches": None,
}


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML or remove --config.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of option names to values.")
    if "learning_rate" in data and "lr" not in data:
        data["lr"] = data.pop("learning_rate")
    return data


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ANN-to-SNN structural evolution stages on MNIST."
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4], default=defaults["stage"])
    parser.add_argument("--dataset", choices=["mnist"], default=defaults["dataset"])
    parser.add_argument("--data-dir", type=Path, default=Path(defaults["data_dir"]))
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--time-steps", type=int, default=defaults["time_steps"])
    parser.add_argument("--threshold", type=float, default=defaults["threshold"])
    parser.add_argument("--hidden-dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--output-dir", type=Path, default=Path(defaults["output_dir"]))
    parser.add_argument("--metrics-csv", type=Path, default=defaults["metrics_csv"])
    parser.add_argument("--device", default=defaults["device"], help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--download", dest="download", action="store_true", default=defaults["download"])
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=defaults["dry_run"])
    parser.add_argument("--max-train-batches", type=int, default=defaults["max_train_batches"])
    parser.add_argument("--max-test-batches", type=int, default=defaults["max_test_batches"])
    return parser


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    known, _ = bootstrap.parse_known_args()

    defaults = DEFAULTS.copy()
    defaults.update(_load_config(known.config))
    return _build_parser(defaults).parse_args()


def _checkpoint_config(args: argparse.Namespace) -> dict:
    config = vars(args).copy()
    for key in ("config", "data_dir", "output_dir", "metrics_csv"):
        value = config.get(key)
        if value is not None:
            config[key] = str(value)
    return config


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    seed_all(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.metrics_csv or output_dir / f"stage{args.stage}_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    model = get_model(
        stage=args.stage,
        hidden_dim=args.hidden_dim,
        time_steps=args.time_steps,
        threshold=args.threshold,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=2e-4,
    )

    train_loader, test_loader, train_size, test_size = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        pin_memory=device.type == "cuda",
    )

    parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"device: {device}")
    print(f"stage: {args.stage} | parameters: {parameters:,}")
    print(f"train samples: {train_size:,} | test samples: {test_size:,}")
    print(f"metrics csv: {metrics_csv}")

    best_accuracy = 0.0
    max_train_batches = 1 if args.dry_run else args.max_train_batches
    max_test_batches = 1 if args.dry_run else args.max_test_batches
    checkpoint_path = output_dir / f"stage{args.stage}_best.pt"

    with metrics_csv.open("w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "stage",
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
                "best_accuracy",
            ],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                max_batches=max_train_batches,
            )
            scheduler.step()
            test_metrics = evaluate(
                model,
                test_loader,
                loss_fn,
                device,
                max_batches=max_test_batches,
            )

            if test_metrics["accuracy"] > best_accuracy:
                best_accuracy = test_metrics["accuracy"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_accuracy": best_accuracy,
                        "config": _checkpoint_config(args),
                    },
                    checkpoint_path,
                )

            writer.writerow(
                {
                    "stage": args.stage,
                    "epoch": epoch,
                    "learning_rate": lr,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "best_accuracy": best_accuracy,
                }
            )
            metrics_file.flush()

            print(
                f"epoch {epoch:03d}/{epochs:03d} | "
                f"lr {lr:.6f} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train acc {train_metrics['accuracy']:.4f} | "
                f"test loss {test_metrics['loss']:.4f} | "
                f"test acc {test_metrics['accuracy']:.4f} | "
                f"best {best_accuracy:.4f}"
            )

    print(f"best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
