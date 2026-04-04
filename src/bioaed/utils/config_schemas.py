"""Pydantic V2 configuration schemas for strict runtime validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class DatasetConfig(BaseModel):
    """Schema for dataset configuration."""

    name: Literal["aswine", "anuraset"]
    root_dir: str
    sample_rate: PositiveInt
    segment_duration: PositiveFloat
    n_mels: PositiveInt = 64
    hop_length: PositiveInt
    win_length: PositiveInt
    num_classes: PositiveInt
    batch_size: PositiveInt = 64
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    normalize: bool = False
    # aSwine-specific
    meta_variant: Literal["raw", "1s", "1s_pruned"] | None = None


class ModelConfig(BaseModel):
    """Schema for model configuration."""

    _target_: str
    num_classes: PositiveInt

    model_config = {"extra": "allow"}


class TrainingConfig(BaseModel):
    """Schema for training configuration."""

    max_epochs: PositiveInt = 100
    learning_rate: PositiveFloat = 1e-3
    weight_decay: float = Field(default=1e-2, ge=0.0)
    patience: PositiveInt = 10
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "bf16-mixed"
    gradient_clip_val: PositiveFloat = 1.0
    accumulate_grad_batches: PositiveInt = 1
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = Field(default=5, ge=0)
    compile: bool = False


class QualityGateConfig(BaseModel):
    """Schema for data-centric quality gate configuration."""

    enabled: bool = True
    snr_threshold: float = 0.0
    spectral_flatness_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    weighting_strategy: Literal["soft", "hard"] = "soft"
    alpha: float = Field(default=0.5, gt=0.0)
    beta: float = Field(default=10.0, gt=0.0)


class AugmentationConfig(BaseModel):
    """Schema for data augmentation configuration."""

    enabled: bool = True
    freq_mask_param: PositiveInt = 8
    time_mask_param: PositiveInt = 20
    n_freq_masks: PositiveInt = 2
    n_time_masks: PositiveInt = 2


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration combining all sub-configs."""

    seed: int = 42
    experiment_name: str = "default"
    dataset: DatasetConfig
    training: TrainingConfig
    quality_gate: QualityGateConfig
    augmentation: AugmentationConfig = AugmentationConfig()

    model_config = {"extra": "allow"}
