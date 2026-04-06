"""Tests for ablation study helper functions (no Hydra required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from bioaed.ablation import _build_output_dir, _resolve_model_key, _train_and_evaluate
from bioaed.models.inceptiontime import InceptionTime


class TestResolveModelKey:
    def test_inceptiontime(self) -> None:
        cfg = OmegaConf.create({"model": {"_target_": "bioaed.models.inceptiontime.InceptionTime"}})
        assert _resolve_model_key(cfg) == "inceptiontime"

    def test_ast(self) -> None:
        cfg = OmegaConf.create(
            {"model": {"_target_": "bioaed.models.ast_model.AudioSpectrogramTransformer"}}
        )
        assert _resolve_model_key(cfg) == "audiospectrogramtransformer"

    def test_missing_target(self) -> None:
        cfg = OmegaConf.create({"model": {"num_classes": 7}})
        assert _resolve_model_key(cfg) == "inceptiontime"


class TestBuildOutputDir:
    def test_dcqg_on_path(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "aswine"},
                "quality_gate": {"enabled": True},
                "seed": 42,
            }
        )
        out = _build_output_dir(cfg, "inceptiontime")
        assert "dcqg_on" in str(out)
        assert "seed_42" in str(out)

    def test_dcqg_off_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "aswine"},
                "quality_gate": {"enabled": False},
                "seed": 0,
            }
        )
        out = _build_output_dir(cfg, "inceptiontime")
        assert "dcqg_off" in str(out)

    def test_fold_included_in_path(self) -> None:
        cfg = OmegaConf.create(
            {
                "dataset": {"name": "anuraset"},
                "quality_gate": {"enabled": True},
                "seed": 1,
            }
        )
        out = _build_output_dir(cfg, "ast", fold=2)
        assert "fold_2" in str(out)


class _TinyDataset(Dataset):
    """Minimal in-memory dataset for mock-based training tests."""

    def __init__(self, n: int = 4, num_classes: int = 7) -> None:
        self.n = n
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = torch.randn(64, 100)
        labels = torch.randint(0, 2, (self.num_classes,)).float()
        weight = torch.tensor(1.0)
        return spec, labels, weight


class TestTrainAndEvaluate:
    """Mock-based tests for _train_and_evaluate — covers lines 57-97 in ablation.py."""

    @pytest.fixture
    def minimal_cfg(self) -> OmegaConf:
        return OmegaConf.create(
            {
                "seed": 42,
                "dataset": {"name": "aswine"},
                "model": {
                    "_target_": "bioaed.models.inceptiontime.InceptionTime",
                    "num_classes": 7,
                    "in_channels": 64,
                    "depth": 1,
                    "n_filters": 16,
                },
                "training": {
                    "max_epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-2,
                    "patience": 5,
                    "precision": "32",
                    "gradient_clip_val": 1.0,
                    "accumulate_grad_batches": 1,
                    "scheduler": "cosine",
                    "warmup_epochs": 0,
                    "compile": False,
                },
                "quality_gate": {
                    "enabled": False,
                    "snr_threshold": 0.0,
                    "spectral_flatness_threshold": 0.0,
                    "weighting_strategy": "soft",
                },
            }
        )

    def test_returns_expected_keys(self, minimal_cfg: OmegaConf, tmp_path: Path) -> None:
        """_train_and_evaluate must return val_metrics, test_metrics, and config."""
        loader = DataLoader(_TinyDataset(n=4, num_classes=7), batch_size=4)

        mock_dm = MagicMock()
        mock_dm.train_dataloader.return_value = loader
        mock_dm.val_dataloader.return_value = loader
        mock_dm.test_dataloader.return_value = loader

        with (
            patch("bioaed.ablation.Fabric") as mock_fabric_cls,
            patch("bioaed.ablation.FabricTrainer") as mock_trainer_cls,
        ):
            mock_fabric = mock_fabric_cls.return_value
            mock_fabric.device = torch.device("cpu")

            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.fit.return_value = {"val_loss": 0.42, "mAP": 0.55}

            result = _train_and_evaluate(minimal_cfg, mock_dm, tmp_path)

        assert "val_metrics" in result
        assert "test_metrics" in result
        assert "config" in result
        assert result["val_metrics"]["val_loss"] == pytest.approx(0.42)
        # test_metrics are computed from real InceptionTime forward passes
        assert "mAP" in result["test_metrics"]

    def test_loads_checkpoint_when_present(self, minimal_cfg: OmegaConf, tmp_path: Path) -> None:
        """If checkpoint_best.pt exists, the model state must be reloaded."""
        loader = DataLoader(_TinyDataset(n=4, num_classes=7), batch_size=4)

        # Create a dummy checkpoint
        model = InceptionTime(num_classes=7, in_channels=64, depth=1, n_filters=16)
        ckpt_path = tmp_path / "checkpoint_best.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        mock_dm = MagicMock()
        mock_dm.train_dataloader.return_value = loader
        mock_dm.val_dataloader.return_value = loader
        mock_dm.test_dataloader.return_value = loader

        with (
            patch("bioaed.ablation.Fabric") as mock_fabric_cls,
            patch("bioaed.ablation.FabricTrainer") as mock_trainer_cls,
        ):
            mock_fabric = mock_fabric_cls.return_value
            mock_fabric.device = torch.device("cpu")
            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.fit.return_value = {"val_loss": 0.3}

            result = _train_and_evaluate(minimal_cfg, mock_dm, tmp_path)

        assert "val_metrics" in result

