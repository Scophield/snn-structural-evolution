"""Generate README figures from training logs and checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from snn_structural_evolution.data import get_mnist_loaders
from snn_structural_evolution.stages import get_model
from snn_structural_evolution.training import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence figures for the README.")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--metrics", nargs="*", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/stage4_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--no-download", dest="download", action="store_false", default=True)
    return parser.parse_args()


def save_structural_evolution(output_dir: Path) -> Path:
    labels = [
        ("Stage 0", "ANN baseline\nReLU activation"),
        ("Stage 1", "Binarized ANN\nthreshold activation"),
        ("Stage 2", "Temporal SNN\ninput expansion"),
        ("Stage 3", "Accumulating SNN\nmembrane integration"),
        ("Stage 4", "Reset SNN\nsparse events"),
    ]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#ECA82C", "#E45756"]

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_axis_off()
    x_positions = np.linspace(0.08, 0.92, len(labels))

    for index, ((stage, text), color) in enumerate(zip(labels, colors)):
        ax.text(
            x_positions[index],
            0.58,
            f"{stage}\n{text}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox={
                "boxstyle": "round,pad=0.45,rounding_size=0.08",
                "facecolor": color,
                "edgecolor": "none",
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.08, 0.58),
                xytext=(x_positions[index] + 0.08, 0.58),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#333333"},
            )

    ax.text(
        0.5,
        0.12,
        "A progressive path from dense ANN computation to sparse event-driven behavior",
        ha="center",
        va="center",
        fontsize=12,
        color="#333333",
    )

    output_path = output_dir / "structural_evolution.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_metrics(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["source"] = path.name
                rows.append(row)
    return rows


def save_metrics_plot(output_dir: Path, metrics_paths: list[Path]) -> Path:
    rows = load_metrics(metrics_paths)
    output_path = output_dir / "training_curves.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No metrics CSV found yet.\nRun train.py, then rerun scripts/make_figures.py.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
    else:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(str(row["stage"]), []).append(row)

        for stage, stage_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
            stage_rows = sorted(stage_rows, key=lambda row: int(row["epoch"]))
            epochs = [int(row["epoch"]) for row in stage_rows]
            train_acc = [float(row["train_accuracy"]) for row in stage_rows]
            test_acc = [float(row["test_accuracy"]) for row in stage_rows]
            ax.plot(epochs, test_acc, marker="o", linewidth=2, label=f"Stage {stage} test")
            ax.plot(epochs, train_acc, linestyle="--", alpha=0.55, label=f"Stage {stage} train")

        ax.set_title("Training and Test Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    stage = int(config.get("stage", 4))
    model = get_model(
        stage=stage,
        hidden_dim=int(config.get("hidden_dim", 10)),
        time_steps=int(config.get("time_steps", 4)),
        threshold=float(config.get("threshold", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def collect_predictions(model, data_loader, device: torch.device, max_batches: int):
    all_targets = []
    all_predictions = []
    first_images = None
    first_targets = None
    first_predictions = None

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(data_loader):
            if batch_index >= max_batches:
                break
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()
            all_targets.append(targets)
            all_predictions.append(predictions)
            if first_images is None:
                first_images = images.detach().cpu()
                first_targets = targets.detach().cpu()
                first_predictions = predictions.detach().cpu()

    return (
        torch.cat(all_targets),
        torch.cat(all_predictions),
        first_images,
        first_targets,
        first_predictions,
    )


def save_confusion_matrix(
    output_dir: Path,
    model,
    data_loader,
    device: torch.device,
    max_batches: int,
) -> Path:
    targets, predictions, _, _, _ = collect_predictions(model, data_loader, device, max_batches)
    matrix = np.zeros((10, 10), dtype=np.int64)
    for target, prediction in zip(targets.numpy(), predictions.numpy()):
        matrix[target, prediction] += 1

    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix on {len(targets)} Test Samples")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for row in range(10):
        for col in range(10):
            value = matrix[row, col]
            if value:
                ax.text(col, row, str(value), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    output_path = output_dir / "stage4_confusion_matrix.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_spike_activity(
    output_dir: Path,
    model,
    data_loader,
    device: torch.device,
    max_batches: int,
) -> Path:
    if not hasattr(model, "spike_sequence"):
        raise RuntimeError("Selected checkpoint model does not expose spike_sequence().")

    best = None
    best_score = float("inf")
    fallback = None
    fallback_score = float("inf")

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(data_loader):
            if batch_index >= max_batches:
                break

            batch = images.to(device)
            spike_seq = model.spike_sequence(batch).detach().cpu()
            predictions = model(batch).argmax(dim=1).cpu()
            spike_counts = spike_seq.sum(dim=(0, 2))

            for sample_index in range(images.shape[0]):
                score = float(spike_counts[sample_index].item())
                if score <= 0:
                    continue

                sample = (
                    images[sample_index].detach().cpu(),
                    int(targets[sample_index].item()),
                    int(predictions[sample_index].item()),
                    spike_seq[:, sample_index, :],
                )

                if score < fallback_score:
                    fallback = sample
                    fallback_score = score

                if predictions[sample_index].item() == targets[sample_index].item() and score < best_score:
                    best = sample
                    best_score = score

    if best is None:
        best = fallback

    if best is None:
        with torch.no_grad():
            images, targets = next(iter(data_loader))
            batch = images[:1].to(device)
            spike_seq = model.spike_sequence(batch).detach().cpu()
            predictions = model(batch).argmax(dim=1).cpu()
            best = (
                images[0].detach().cpu(),
                int(targets[0].item()),
                int(predictions[0].item()),
                spike_seq[:, 0, :],
            )

    image, target, prediction, spikes = best

    firing_rate = spikes.mean(dim=0).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1, 1.6, 1.2]})
    axes[0].imshow(image[0], cmap="gray")
    axes[0].set_title(f"MNIST input\ntrue {target}, pred {prediction}")
    axes[0].set_axis_off()

    axes[1].imshow(spikes.T.numpy(), aspect="auto", cmap="Greys")
    axes[1].set_title(f"Spike Raster\n{int(spikes.sum().item())} total spikes")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Hidden neuron")
    axes[1].set_xticks(range(spikes.shape[0]))

    axes[2].bar(range(len(firing_rate)), firing_rate, color="#E45756")
    axes[2].set_title("Neuron Firing Rate")
    axes[2].set_xlabel("Hidden neuron")
    axes[2].set_ylabel("Rate")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.25)

    output_path = output_dir / "stage4_spike_activity.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        save_structural_evolution(args.output_dir),
        save_metrics_plot(
            args.output_dir,
            args.metrics if args.metrics is not None else sorted(Path("runs").glob("stage*_metrics.csv")),
        ),
    ]

    if args.checkpoint.exists():
        device = resolve_device(args.device)
        model, _ = load_checkpoint_model(args.checkpoint, device)
        _, test_loader, _, _ = get_mnist_loaders(
            data_dir=args.data_dir,
            batch_size=64,
            download=args.download,
            pin_memory=device.type == "cuda",
        )
        generated.append(save_confusion_matrix(args.output_dir, model, test_loader, device, args.max_batches))
        generated.append(save_spike_activity(args.output_dir, model, test_loader, device, args.max_batches))
    else:
        print(f"Skipping checkpoint figures because {args.checkpoint} was not found.")

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
