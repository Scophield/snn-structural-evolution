"""Surrogate gradients for spike activations."""

from __future__ import annotations

import torch


class SigmoidSurrogate(torch.autograd.Function):
    """Binary spike in the forward pass, sigmoid derivative in backward."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.alpha = alpha
        return (inputs > 0).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (inputs,) = ctx.saved_tensors
        alpha = ctx.alpha
        sigma = torch.sigmoid(inputs * alpha)
        grad_inputs = grad_output * (1.0 - sigma) * sigma * alpha
        return grad_inputs, None


def spike_function(inputs: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Apply the default binary spike surrogate."""
    return SigmoidSurrogate.apply(inputs, alpha)
