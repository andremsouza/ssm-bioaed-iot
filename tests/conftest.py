"""Shared pytest fixtures for BioAED tests."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def dummy_spectrogram() -> torch.Tensor:
    """Create a dummy log-mel spectrogram tensor (batch=4, n_mels=64, time=100)."""
    return torch.randn(4, 64, 100)


@pytest.fixture
def dummy_labels_7() -> torch.Tensor:
    """Create dummy multi-hot labels for 7 classes (aSwine)."""
    return torch.randint(0, 2, (4, 7)).float()


@pytest.fixture
def dummy_labels_42() -> torch.Tensor:
    """Create dummy multi-hot labels for 42 classes (AnuraSet)."""
    return torch.randint(0, 2, (4, 42)).float()


@pytest.fixture
def aswine_cfg() -> OmegaConf:
    """Create a minimal aSwine experiment config."""
    return OmegaConf.create(
        {
            "seed": 42,
            "experiment_name": "test",
            "dataset": {
                "name": "aswine",
                "root_dir": "data/aswine",
                "meta_variant": "1s_pruned",
                "sample_rate": 16000,
                "segment_duration": 1.0,
                "n_mels": 64,
                "hop_length": 160,
                "win_length": 400,
                "num_classes": 7,
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
            },
            "model": {
                "_target_": "bioaed.models.inceptiontime.InceptionTime",
                "num_classes": 7,
                "in_channels": 64,
                "depth": 1,
                "n_filters": 16,
            },
            "training": {
                "max_epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "patience": 5,
                "precision": "32",
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "scheduler": "cosine",
                "warmup_epochs": 0,
            },
        }
    )
