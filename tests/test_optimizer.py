"""Tests for optimizer and scheduler factory."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR

from bioaed.training.optimizer import build_optimizer_and_scheduler


def _make_model() -> nn.Module:
    return nn.Linear(10, 2)


class TestBuildOptimizerAndScheduler:
    def test_cosine_scheduler(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            learning_rate=1e-3,
            weight_decay=1e-2,
            max_epochs=100,
            scheduler_name="cosine",
            warmup_epochs=0,
        )
        assert isinstance(opt, torch.optim.AdamW)
        assert sched is not None

    def test_step_scheduler(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            scheduler_name="step",
            warmup_epochs=0,
        )
        assert sched is not None

    def test_no_scheduler(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            scheduler_name="none",
            warmup_epochs=0,
        )
        assert sched is None

    def test_warmup_creates_sequential(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            scheduler_name="cosine",
            warmup_epochs=5,
            max_epochs=50,
        )
        assert isinstance(sched, SequentialLR)

    def test_warmup_with_step_scheduler(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            scheduler_name="step",
            warmup_epochs=3,
        )
        assert isinstance(sched, SequentialLR)

    def test_warmup_with_none_scheduler(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            scheduler_name="none",
            warmup_epochs=5,
        )
        assert sched is None

    def test_lr_set_correctly(self) -> None:
        model = _make_model()
        opt, _ = build_optimizer_and_scheduler(
            model,
            learning_rate=0.01,
            weight_decay=0.05,
            warmup_epochs=0,
        )
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[0]["weight_decay"] == 0.05

    def test_scheduler_step_changes_lr(self) -> None:
        model = _make_model()
        opt, sched = build_optimizer_and_scheduler(
            model,
            learning_rate=1e-3,
            scheduler_name="cosine",
            warmup_epochs=0,
            max_epochs=10,
        )
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(5):
            sched.step()
        assert opt.param_groups[0]["lr"] != initial_lr
