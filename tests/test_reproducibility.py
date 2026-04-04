"""Tests for reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from bioaed.utils.reproducibility import seed_everything


class TestSeedEverything:
    def test_python_random_seeded(self) -> None:
        seed_everything(123)
        a = random.random()
        seed_everything(123)
        b = random.random()
        assert a == b

    def test_numpy_seeded(self) -> None:
        seed_everything(99)
        a = np.random.rand()
        seed_everything(99)
        b = np.random.rand()
        assert a == b

    def test_torch_seeded(self) -> None:
        seed_everything(7)
        a = torch.randn(10)
        seed_everything(7)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_env_hash_seed(self) -> None:
        seed_everything(42)
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_cudnn_flags(self) -> None:
        seed_everything(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
