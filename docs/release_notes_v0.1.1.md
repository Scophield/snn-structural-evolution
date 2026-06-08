# v0.1.1: Full MNIST Benchmark Results

This release updates the project with a full Stage 0-4 MNIST benchmark using the release-oriented configuration.

## Benchmark Settings

- `hidden_dim=128`
- `epochs=100`
- full MNIST train/test batches
- seed 35
- 4 time steps for temporal stages

## Environment

- Python 3.9.13
- PyTorch 1.12.0
- torchvision 0.13.0
- CUDA available
- GPU: Quadro P620
- Runtime: about 1 h 41 min

## Results

| Stage | Train accuracy | Test accuracy | Spike rate | Sparsity | Event ops proxy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 99.80% | 97.91% | - | - | - |
| 1 | 99.24% | 97.09% | - | - | - |
| 2 | 99.22% | 97.27% | 34.83% | 65.17% | 17,833,800 |
| 3 | 99.06% | 97.41% | 40.00% | 60.00% | 20,481,800 |
| 4 | 99.25% | 97.24% | 25.79% | 74.21% | 13,206,980 |

Stage 4 keeps high MNIST accuracy while producing the sparsest event activity among the temporal stages.
