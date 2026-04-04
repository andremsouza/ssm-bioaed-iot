"""Tests for Optuna HPO suggest functions and sweep helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf

from bioaed.hpo.optuna_sweep import (
    _compute_pos_weight,
    _suggest_common,
    _suggest_dcqg,
    _suggest_model_specific,
    run_sweep,
)


def _base_cfg_dict() -> dict:
    return {
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
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "_target_": "bioaed.models.inceptiontime.InceptionTime",
            "num_classes": 7,
            "in_channels": 64,
        },
        "training": {
            "max_epochs": 5,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "patience": 3,
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
            "alpha": 0.5,
            "beta": 10.0,
        },
        "augmentation": {
            "enabled": True,
            "freq_mask_param": 8,
            "time_mask_param": 20,
            "n_freq_masks": 2,
            "n_time_masks": 2,
        },
    }


class TestSuggestCommon:
    def test_modifies_training(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 1e-4,
                "weight_decay": 1e-3,
                "batch_size": 64,
                "scheduler": "cosine",
                "warmup_epochs": 3,
                "gradient_clip_val": 2.0,
                "use_pos_weight": True,
                "augmentation_enabled": True,
                "freq_mask_param": 12,
                "time_mask_param": 30,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_common(trial, cfg)

        assert cfg["training"]["learning_rate"] == 1e-4
        assert cfg["training"]["weight_decay"] == 1e-3
        assert cfg["training"]["batch_size"] == 64
        assert cfg["dataset"]["batch_size"] == 64
        assert cfg["training"]["scheduler"] == "cosine"
        assert cfg["training"]["warmup_epochs"] == 3
        assert cfg["augmentation"]["enabled"] is True

    def test_augmentation_disabled(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 1e-3,
                "weight_decay": 1e-2,
                "batch_size": 32,
                "scheduler": "step",
                "warmup_epochs": 0,
                "gradient_clip_val": 1.0,
                "use_pos_weight": False,
                "augmentation_enabled": False,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_common(trial, cfg)
        assert cfg["augmentation"]["enabled"] is False


class TestSuggestDcqg:
    def test_enabled(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "snr_threshold": 5.0,
                "spectral_flatness_threshold": 0.3,
                "alpha": 1.0,
                "beta": 15.0,
            }
        )
        cfg = _base_cfg_dict()
        cfg["quality_gate"]["enabled"] = True
        _suggest_dcqg(trial, cfg)
        assert cfg["quality_gate"]["snr_threshold"] == 5.0
        assert cfg["quality_gate"]["alpha"] == 1.0

    def test_disabled_no_changes(self) -> None:
        trial = optuna.trial.FixedTrial({})
        cfg = _base_cfg_dict()
        cfg["quality_gate"]["enabled"] = False
        _suggest_dcqg(trial, cfg)
        assert cfg["quality_gate"]["snr_threshold"] == 0.0  # unchanged


class TestSuggestModelSpecific:
    def test_inceptiontime(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "depth": 3,
                "n_filters": 32,
                "kernel_sizes": "medium",
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "inceptiontime")
        assert cfg["model"]["depth"] == 3
        assert cfg["model"]["n_filters"] == 32
        assert cfg["model"]["kernel_sizes"] == [10, 20, 40]

    def test_ast(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "model_name": "vit_base_patch16_224",
                "pretrained": False,
                "learning_rate": 2e-5,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "ast")
        assert cfg["model"]["model_name"] == "vit_base_patch16_224"

    def test_audio_mamba(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "d_model": 128,
                "n_layers": 4,
                "d_state": 16,
                "patch_size": 16,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "audio_mamba")
        assert cfg["model"]["d_model"] == 128
        assert cfg["model"]["n_layers"] == 4

    def test_unknown_model_no_error(self) -> None:
        trial = optuna.trial.FixedTrial({})
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "unknown")  # should not raise


class TestComputePosWeight:
    def test_none_when_no_train_dataset(self) -> None:
        dm = MagicMock()
        dm.train_dataset = None
        result = _compute_pos_weight(dm)
        assert result is None

    def test_returns_tensor(self) -> None:
        dm = MagicMock()
        # Simulate Subset with metadata
        metadata = [{"labels": [1, 0, 1]} for _ in range(8)] + [
            {"labels": [0, 1, 0]} for _ in range(4)
        ]
        dm.train_dataset.dataset.metadata = metadata
        dm.train_dataset.indices = list(range(12))
        result = _compute_pos_weight(dm)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)


class TestRunSweep:
    def test_returns_study(self, tmp_path, monkeypatch) -> None:
        """Test run_sweep with mocked objective to avoid real training."""
        monkeypatch.setattr(
            "bioaed.hpo.optuna_sweep.objective",
            lambda trial, cfg: 0.5,
        )
        cfg = OmegaConf.create(_base_cfg_dict())
        study = run_sweep(cfg, n_trials=2, n_startup_trials=1)
        assert isinstance(study, optuna.Study)
        assert study.best_value == 0.5
        assert len(study.trials) == 2
