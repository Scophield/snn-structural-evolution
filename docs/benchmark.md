# Benchmark Notes

The checked-in figures are generated from a short Stage 4 run so users can see the expected artifact format immediately.

## Preliminary v0.1 Sanity Benchmark

This table uses 5 epochs, 120 training batches per epoch, and 80 test batches on MNIST with seed 35.

| Stage | Train accuracy | Test accuracy | Spike rate | Sparsity | Event ops proxy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 91.15% | 89.36% | - | - | - |
| 1 | 87.04% | 83.38% | - | - | - |
| 2 | 87.15% | 82.21% | 43.53% | 56.47% | 891,480 |
| 3 | 87.19% | 84.00% | 49.41% | 50.59% | 1,011,900 |
| 4 | 87.98% | 84.06% | 43.56% | 56.44% | 892,150 |

Reproduce the preliminary table with:

```bash
python examples/compare_stages.py --epochs 5 --max-train-batches 120 --max-test-batches 80 --data-dir ./data
```

For release-quality numbers, run full-stage benchmarks with a fixed environment:

```bash
python examples/compare_stages.py --epochs 100 --max-train-batches 938 --max-test-batches 157
```

Recommended metrics:

| Metric | Why it matters |
| --- | --- |
| Accuracy | Classification quality |
| Spike rate | Event density |
| Activation sparsity | Hardware-friendly inactivity |
| Event ops proxy | Approximate event-driven readout cost |
| Accuracy drop | Cost of structural conversion |

Report Python, PyTorch, CUDA, GPU/CPU, random seed, epochs, batch size, learning rate, time steps, and threshold with every benchmark table.
