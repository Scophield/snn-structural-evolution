"""ANN-to-SNN structural evolution stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from snn_structural_evolution.models import TwoLayerMNISTBackbone
from snn_structural_evolution.surrogate import spike_function


class Stage0ANN(TwoLayerMNISTBackbone):
    """Baseline ANN with a ReLU hidden activation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.lin1(self.flatten_images(inputs)))
        return self.lin2(hidden)


class Stage1BinarizedANN(TwoLayerMNISTBackbone):
    """ANN with thresholded binary hidden activations."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 10,
        num_classes: int = 10,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.lin1(self.flatten_images(inputs))
        spikes = spike_function(hidden - self.threshold, self.surrogate_alpha)
        return self.lin2(spikes)


class TemporalStage(TwoLayerMNISTBackbone):
    """Base class for stages that repeat inputs over simulation time."""

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 10,
        num_classes: int = 10,
        time_steps: int = 4,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
    ) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
        if time_steps < 1:
            raise ValueError("time_steps must be at least 1")
        self.time_steps = time_steps
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha

    def expand_time(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

    def hidden_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal_inputs = self.expand_time(inputs)
        return self.lin1(self.flatten_temporal_images(temporal_inputs))

    def readout(self, spikes: torch.Tensor) -> torch.Tensor:
        logits = self.lin2(spikes)
        return logits.mean(dim=0)


class Stage2TemporalSNN(TemporalStage):
    """Direct temporal expansion with independent thresholding at each step."""

    def spike_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_seq = self.hidden_sequence(inputs)
        return spike_function(hidden_seq - self.threshold, self.surrogate_alpha)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.readout(self.spike_sequence(inputs))


class Stage3AccumulatingSNN(TemporalStage):
    """Integrate-and-fire style membrane accumulation without reset."""

    def spike_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_seq = self.hidden_sequence(inputs)
        membrane = torch.zeros_like(hidden_seq[0])
        spike_seq = []

        for step_inputs in hidden_seq:
            membrane = membrane + step_inputs
            spikes = spike_function(membrane - self.threshold, self.surrogate_alpha)
            spike_seq.append(spikes)

        return torch.stack(spike_seq, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.readout(self.spike_sequence(inputs))


class Stage4ResetSNN(TemporalStage):
    """Event-driven SNN with membrane accumulation and reset after spikes."""

    def spike_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_seq = self.hidden_sequence(inputs)
        membrane = torch.zeros_like(hidden_seq[0])
        spike_seq = []

        for step_inputs in hidden_seq:
            membrane = membrane + step_inputs
            spikes = spike_function(membrane - self.threshold, self.surrogate_alpha)
            spike_seq.append(spikes)
            membrane = membrane - spikes * self.threshold

        return torch.stack(spike_seq, dim=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.readout(self.spike_sequence(inputs))


STAGE_REGISTRY = {
    0: Stage0ANN,
    1: Stage1BinarizedANN,
    2: Stage2TemporalSNN,
    3: Stage3AccumulatingSNN,
    4: Stage4ResetSNN,
}


def get_model(
    stage: int,
    input_dim: int = 28 * 28,
    hidden_dim: int = 10,
    num_classes: int = 10,
    time_steps: int = 4,
    threshold: float = 1.0,
    surrogate_alpha: float = 4.0,
):
    """Build a model for one structural evolution stage."""
    try:
        stage_id = int(stage)
        model_cls = STAGE_REGISTRY[stage_id]
    except (KeyError, ValueError) as exc:
        valid = ", ".join(str(key) for key in sorted(STAGE_REGISTRY))
        raise ValueError(f"stage must be one of: {valid}") from exc

    if stage_id == 0:
        return model_cls(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    if stage_id == 1:
        return model_cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            threshold=threshold,
            surrogate_alpha=surrogate_alpha,
        )
    return model_cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        time_steps=time_steps,
        threshold=threshold,
        surrogate_alpha=surrogate_alpha,
    )


ANN = Stage0ANN
SNN = Stage4ResetSNN

stage0 = Stage0ANN
stage1 = Stage1BinarizedANN
stage2 = Stage2TemporalSNN
stage3 = Stage3AccumulatingSNN
stage4 = Stage4ResetSNN
