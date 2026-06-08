# Benchmark Notes

The checked-in figures are generated from a short Stage 4 run so users can see the expected artifact format immediately.

## Full MNIST Benchmark

This table uses 100 epochs, full MNIST train/test batches, `hidden_dim=128`, seed 35, and 4 time steps for temporal stages.

Environment used for this run:

| Item | Value |
| --- | --- |
| Python | 3.9.13 |
| PyTorch | 1.12.0 |
| CUDA | available |
| GPU | Quadro P620 |
| Runtime | about 1 h 41 min |

| Stage | Train accuracy | Test accuracy | Spike rate | Sparsity | Event ops proxy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 99.80% | 97.91% | - | - | - |
| 1 | 99.24% | 97.09% | - | - | - |
| 2 | 99.22% | 97.27% | 34.83% | 65.17% | 17,833,800 |
| 3 | 99.06% | 97.41% | 40.00% | 60.00% | 20,481,800 |
| 4 | 99.25% | 97.24% | 25.79% | 74.21% | 13,206,980 |

Raw CSV: [results/mnist_release_hidden128_full_20260608.csv](results/mnist_release_hidden128_full_20260608.csv)

Reproduce the full-stage benchmark with:

```bash
python examples/compare_stages.py --epochs 100 --hidden-dim 128 --max-train-batches 938 --max-test-batches 157
```

Use `configs/mnist_tiny.yaml` for teaching and `configs/mnist_release.yaml` as the default accuracy-oriented benchmark configuration.

Recommended metrics:

| Metric | Why it matters |
| --- | --- |
| Accuracy | Classification quality |
| Spike rate | Event density |
| Activation sparsity | Hardware-friendly inactivity |
| Event ops proxy | Approximate event-driven readout cost |
| Accuracy drop | Cost of structural conversion |

Report Python, PyTorch, CUDA, GPU/CPU, random seed, epochs, batch size, learning rate, time steps, and threshold with every benchmark table.
