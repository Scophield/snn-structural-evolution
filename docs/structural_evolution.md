# Structural Evolution

This toolkit frames ANN-to-SNN conversion as a staged structural evolution process rather than a single replacement step.

## Stage 0: ANN Baseline

Stage 0 is a dense ANN baseline with a ReLU hidden activation. It anchors accuracy and parameter count.

## Stage 1: Activation Binarization

Stage 1 replaces continuous hidden activations with binary threshold activations trained through a surrogate gradient.

## Stage 2: Temporal Expansion

Stage 2 repeats inputs over simulation time, exposing the model to temporal computation while keeping thresholding independent at each step.

## Stage 3: Membrane Accumulation

Stage 3 introduces integrate-and-fire style membrane accumulation, allowing state to persist across time steps.

## Stage 4: Reset-Based Sparse SNN

Stage 4 resets membrane potential after spikes. This creates sparse event-driven behavior and is the main target for neuromorphic and hardware-aware experiments.
