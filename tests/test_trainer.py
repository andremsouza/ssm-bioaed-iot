"""Tests for FabricTrainer with a tiny model and synthetic data."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from bioaed.training.trainer import FabricTrainer


def _tiny_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "training": {
                "max_epochs": 3,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "patience": 10,
                "precision": "32",
                "gradient_clip_val": 1.0,
                "accumulate_grad_batches": 1,
                "scheduler": "none",
                "warmup_epochs": 0,
                "compile": False,
            }
        }
    )


def _make_loader(
    n_samples: int = 16, n_features: int = 10, n_classes: int = 3, batch_size: int = 8
) -> DataLoader:
    x = torch.randn(n_samples, n_features, 20)
    y = torch.randint(0, 2, (n_samples, n_classes)).float()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TestFabricTrainer:
    def test_fit_returns_metrics(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg()
        fabric = Fabric(accelerator="cpu", precision="32-true")
        model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(10, 3))
        train_loader = _make_loader()
        val_loader = _make_loader(n_samples=8)

        trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=tmp_path)
        result = trainer.fit(model, train_loader, val_loader)

        assert "val_loss" in result
        assert "train_loss" in result
        assert "lr" in result

    def test_saves_checkpoints(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg()
        fabric = Fabric(accelerator="cpu", precision="32-true")
        model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(10, 3))

        trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=tmp_path)
        trainer.fit(model, _make_loader(), _make_loader(n_samples=8))

        assert (tmp_path / "checkpoint_best.pt").exists()
        assert (tmp_path / "checkpoint_last.pt").exists()

    def test_early_stopping_triggers(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg()
        cfg.training.max_epochs = 100
        cfg.training.patience = 1  # stop after 1 non-improving epoch
        fabric = Fabric(accelerator="cpu", precision="32-true")
        model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(10, 3))

        trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=tmp_path)
        result = trainer.fit(model, _make_loader(), _make_loader(n_samples=8))

        # Training should stop well before epoch 100
        assert result["epoch"] < 100

    def test_pos_weight_applied(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg()
        fabric = Fabric(accelerator="cpu", precision="32-true")
        model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(10, 3))
        pos_weight = torch.tensor([2.0, 1.0, 0.5])

        trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=tmp_path, pos_weight=pos_weight)
        result = trainer.fit(model, _make_loader(), _make_loader(n_samples=8))
        assert "val_loss" in result

    def test_validate_returns_float(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg()
        fabric = Fabric(accelerator="cpu", precision="32-true")
        model = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(10, 3))
        model, _ = fabric.setup(model, torch.optim.SGD(model.parameters(), lr=0.01))

        criterion = nn.BCEWithLogitsLoss(reduction="none")
        val_loader = fabric.setup_dataloaders(_make_loader(n_samples=8))

        trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=tmp_path)
        val_loss = trainer._validate(model, val_loader, criterion)
        assert isinstance(val_loss, float)
        assert val_loss > 0
