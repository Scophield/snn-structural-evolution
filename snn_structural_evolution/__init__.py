"""Structural evolution from ANN to sparse event-driven SNN."""

from snn_structural_evolution.stages import (
    ANN,
    SNN,
    STAGE_REGISTRY,
    Stage0ANN,
    Stage1BinarizedANN,
    Stage2TemporalSNN,
    Stage3AccumulatingSNN,
    Stage4ResetSNN,
    get_model,
    stage0,
    stage1,
    stage2,
    stage3,
    stage4,
)
from snn_structural_evolution.metrics import accuracy, spike_statistics

__all__ = [
    "ANN",
    "SNN",
    "STAGE_REGISTRY",
    "Stage0ANN",
    "Stage1BinarizedANN",
    "Stage2TemporalSNN",
    "Stage3AccumulatingSNN",
    "Stage4ResetSNN",
    "get_model",
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "accuracy",
    "spike_statistics",
]
