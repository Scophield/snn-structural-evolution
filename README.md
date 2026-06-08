# SNN Structural Evolution Toolkit

[![CI](https://github.com/Scophield/snn-structural-evolution/actions/workflows/ci.yml/badge.svg)](https://github.com/Scophield/snn-structural-evolution/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Scophield/snn-structural-evolution)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

A minimal PyTorch toolkit for structural evolution from ANN to event-driven SNN for hardware-aware neuromorphic computing.

This repository demonstrates how conventional ANN computation can be progressively transformed into sparse, event-driven SNN behavior. The path is designed for researchers working on neuromorphic hardware, compute-in-memory accelerators, ReRAM, and analog AI systems.

![Structural evolution overview](docs/assets/structural_evolution.png)

## Overview

The project is organized around a five-step structural evolution path:

```text
Stage 0: ANN baseline
   |
   v
Stage 1: activation binarization
   |
   v
Stage 2: temporal expansion
   |
   v
Stage 3: membrane accumulation
   |
   v
Stage 4: reset-based sparse event-driven SNN
```

| Stage | Model | Key idea | Time steps |
| ---: | --- | --- | ---: |
| 0 | ANN baseline | ReLU hidden activation | 1 |
| 1 | Binarized ANN | Threshold activation with surrogate gradient | 1 |
| 2 | Temporal SNN | Repeat inputs over simulation time | configurable |
| 3 | Accumulating SNN | Integrate membrane potential over time | configurable |
| 4 | Reset SNN | Reset membrane after spikes for sparse event behavior | configurable |

## Quick Start

Requirements:

- Python 3.10+
- PyTorch 2.0+

```bash
git clone https://github.com/Scophield/snn-structural-evolution.git
cd snn-structural-evolution
python -m pip install -r requirements.txt
python train.py --stage 4 --dataset mnist --epochs 100
```

For a fast smoke run:

```bash
python train.py --stage 4 --epochs 1 --dry-run
```

To compare all stages quickly:

```bash
python examples/compare_stages.py --epochs 1 --max-train-batches 40 --max-test-batches 40
```

## Reproduce MNIST Experiments

Run each stage with the same training settings:

```bash
python train.py --stage 0 --epochs 100 --batch-size 64 --lr 0.005
python train.py --stage 1 --epochs 100 --batch-size 64 --lr 0.005
python train.py --stage 2 --epochs 100 --batch-size 64 --lr 0.005 --time-steps 4
python train.py --stage 3 --epochs 100 --batch-size 64 --lr 0.005 --time-steps 4
python train.py --stage 4 --epochs 100 --batch-size 64 --lr 0.005 --time-steps 4
```

You can also start from the provided config:

```bash
python train.py --config configs/mnist_stage4.yaml
```

The default data path is `./data`, and the default output path is `./runs`. CUDA is used automatically when available; otherwise training falls back to CPU.

## Results on MNIST

The table below is a preliminary short-run example, not a final benchmark. It verifies the pipeline and shows the result format. Regenerate full numbers with `examples/compare_stages.py` before reporting release-quality claims.

| Stage | Description | Accuracy | Time steps | Sparsity | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | ANN baseline | TBD | 1 | - | full benchmark pending |
| 1 | Binarized activation | TBD | 1 | TBD | full benchmark pending |
| 2 | Temporal expansion | TBD | 4 | TBD | full benchmark pending |
| 3 | Membrane accumulation | TBD | 4 | TBD | full benchmark pending |
| 4 | Reset-based sparse SNN | 85.51% | 4 | example-dependent | 5-epoch short run, partial batches |

## Evidence Figures

The repository includes figure-generation tooling for visual evidence:

```bash
python train.py --stage 4 --epochs 100 --metrics-csv runs/stage4_metrics.csv
python scripts/make_figures.py --metrics runs/stage4_metrics.csv --checkpoint runs/stage4_best.pt
```

The checked-in gallery is generated from a short Stage 4 run so the repository has visual evidence immediately. Regenerate it after full training before reporting final benchmark claims.

Training curves:

![Training curves](docs/assets/training_curves.png)

Stage 4 spike activity:

![Stage 4 spike activity](docs/assets/stage4_spike_activity.png)

Stage 4 confusion matrix:

![Stage 4 confusion matrix](docs/assets/stage4_confusion_matrix.png)

## Project Layout

```text
snn-structural-evolution/
|-- train.py
|-- configs/
|   `-- mnist_stage4.yaml
|-- docs/
|   |-- benchmark.md
|   |-- hardware_relevance.md
|   `-- structural_evolution.md
|-- examples/
|   |-- compare_stages.py
|   |-- export_spike_statistics.py
|   `-- run_mnist.py
|-- scripts/
|   `-- make_figures.py
|-- snn_structural_evolution/
|   |-- data.py
|   |-- metrics.py
|   |-- models.py
|   |-- stages.py
|   |-- surrogate.py
|   `-- training.py
`-- tests/
    `-- test_stages.py
```

## Python API

```python
import torch

from snn_structural_evolution import get_model

model = get_model(stage=4, time_steps=4)
images = torch.rand(8, 1, 28, 28)
logits = model(images)
print(logits.shape)
```

## Research Scope

This repository is intentionally small. Its value is to make the ANN-to-SNN transition explicit and reproducible:

- Stage-level comparison under one training loop
- Surrogate-gradient spike activation
- Device-agnostic CPU/CUDA execution
- Reproducible MNIST data path and CLI arguments
- Reset-based sparse event-driven behavior for hardware-aware studies

More background:

- [Structural evolution](docs/structural_evolution.md)
- [Hardware relevance](docs/hardware_relevance.md)
- [Benchmark notes](docs/benchmark.md)
- [FAQ](docs/faq.md)

## Development

Run smoke tests before opening a pull request:

```bash
python -m compileall snn_structural_evolution train.py core.py scripts/make_figures.py
python -m unittest discover tests
```

Legacy source from the initial single-file prototype is kept in `legacy/original_simple_snn_ann.py.txt` for reference.
