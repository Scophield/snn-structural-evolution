# Benchmark Notes

The checked-in figures are generated from a short Stage 4 run so users can see the expected artifact format immediately.

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
