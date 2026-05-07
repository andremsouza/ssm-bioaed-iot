"""Optimizer and learning rate scheduler factory."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)


def build_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    max_epochs: int = 100,
    scheduler_name: Literal["cosine", "step", "none"] = "cosine",
    warmup_epochs: int = 5,
) -> tuple[torch.optim.Optimizer, LRScheduler | None]:
    """Create an AdamW optimizer and optional LR scheduler.

    When ``warmup_epochs > 0`` and a scheduler is selected, a linear warmup
    phase ramps the LR from near-zero to ``learning_rate`` before the main
    schedule takes over via :class:`~torch.optim.lr_scheduler.SequentialLR`.

    Args:
        model: The neural network whose parameters will be optimized.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        max_epochs: Total number of training epochs (for cosine schedule).
        scheduler_name: Type of learning rate scheduler.
        warmup_epochs: Number of linear warmup epochs.

    Returns:
        Tuple of (optimizer, scheduler). Scheduler may be ``None``.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,  # avoids _multi_tensor_adam dtype/device mismatch with mamba-ssm
    )

    scheduler: LRScheduler | None = None
    main_scheduler: LRScheduler | None = None

    if scheduler_name == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-6
        )
    elif scheduler_name == "step":
        main_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    if main_scheduler is not None and warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = main_scheduler

    return optimizer, scheduler
