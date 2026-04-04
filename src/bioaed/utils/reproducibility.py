"""Reproducibility utilities for deterministic training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.multiprocessing as tmp
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

    # Use filesystem-based shared memory for DataLoader workers.
    # The default 'file_descriptor' strategy places shared memory handles in
    # Python's multiprocessing temp dir (/tmp/pymp-*).  On Python 3.13 the
    # temp-dir cleanup finalizer runs before all worker file descriptors are
    # released, causing OSError: [Errno 39] Directory not empty at process exit.
    # 'file_system' uses /dev/shm instead and avoids this race entirely.
    tmp.set_sharing_strategy("file_system")

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
