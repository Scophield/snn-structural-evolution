# Reference Weights

Reference checkpoints are distributed as GitHub Release assets instead of being committed to git.

## Stage 4 Release Checkpoint

| Field | Value |
| --- | --- |
| Stage | 4 |
| Model | Reset-based sparse SNN |
| Dataset | MNIST |
| Hidden dim | 128 |
| Time steps | 4 |
| Epochs | 100 |
| Seed | 35 |
| Best test accuracy | 97.75% |
| Download | [stage4_mnist_hidden128_seed35_v0.1.1.pt](https://github.com/Scophield/snn-structural-evolution/releases/download/v0.1.1/stage4_mnist_hidden128_seed35_v0.1.1.pt) |
| Metrics CSV | [stage4_mnist_hidden128_seed35_v0.1.1_metrics.csv](https://github.com/Scophield/snn-structural-evolution/releases/download/v0.1.1/stage4_mnist_hidden128_seed35_v0.1.1_metrics.csv) |
| SHA256 | `78683332db3c7b1f0c60b5a434e07020daa923c1d0f8ec5579cfd84d153e1d72` |

This checkpoint was trained with:

```bash
python train.py --config configs/mnist_release.yaml
```

It is a standalone Stage 4 reference checkpoint. The Stage 0-4 comparison table in the README is produced by `examples/compare_stages.py`, while this checkpoint is produced by `train.py` with checkpointing enabled.

The checkpoint is saved by `train.py` as a dictionary with:

- `model_state_dict`
- `best_accuracy`
- `config`

Load it with:

```python
import torch

from snn_structural_evolution import get_model

checkpoint = torch.load("stage4_mnist_hidden128_seed35.pt", map_location="cpu")
model = get_model(stage=4, hidden_dim=128, time_steps=4)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(checkpoint["best_accuracy"])
```

Recreate the checkpoint from scratch:

```bash
python train.py --config configs/mnist_release.yaml
```

Verify the downloaded checkpoint:

```bash
Get-FileHash -Algorithm SHA256 stage4_mnist_hidden128_seed35_v0.1.1.pt
```
