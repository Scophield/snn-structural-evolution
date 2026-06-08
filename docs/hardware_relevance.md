# Hardware Relevance

Structural SNN evolution is useful because each stage exposes a hardware-relevant constraint.

| Stage | Hardware-relevant change |
| ---: | --- |
| 0 | Dense ANN compute baseline |
| 1 | Reduced activation precision through binarization |
| 2 | Explicit temporal execution |
| 3 | Stateful membrane accumulation |
| 4 | Sparse event-driven operation through reset |

For compute-in-memory, ReRAM, analog AI, and neuromorphic accelerators, the most important signals are not only accuracy. Useful hardware-facing metrics include spike rate, activation sparsity, event-operation proxy, and ANN-to-SNN accuracy drop.

This repository keeps those metrics explicit so experiments can connect model behavior to hardware-aware design questions.
