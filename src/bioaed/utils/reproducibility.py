"""Reproducibility utilities for deterministic training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility.

    Sets seeds for Python random, NumPy, PyTorch CPU/CUDA, and configures
    cuDNN for deterministic behavior.

    Args:
        seed: The seed value to use across all RNGs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Allow Tensor Cores on Ampere/Ada/Blackwell GPUs (TF32 matmul).
        # 'high' keeps full BF16 accumulation while using TF32 for the inner
        # product — negligible accuracy impact, significant throughput gain.
        torch.set_float32_matmul_precision("high")

    # Deterministic algorithms (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"Seeded all RNGs with seed={seed}")
