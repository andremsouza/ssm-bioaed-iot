"""Tests for Optuna HPO suggest functions and sweep helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import optuna
import torch
from omegaconf import OmegaConf

from bioaed.hpo.optuna_sweep import (
    _compute_pos_weight,
    _suggest_common,
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

    def test_skip_lr_does_not_register_learning_rate(self) -> None:
        """When skip_lr=True, learning_rate must NOT be suggested by _suggest_common
        so that _suggest_model_specific can register it with a narrower range."""
        trial = optuna.trial.FixedTrial(
            {
                "weight_decay": 1e-3,
                "batch_size": 32,
                "scheduler": "cosine",
                "warmup_epochs": 2,
                "gradient_clip_val": 1.0,
                "use_pos_weight": False,
                "augmentation_enabled": False,
            }
        )
        cfg = _base_cfg_dict()
        original_lr = cfg["training"]["learning_rate"]
        _suggest_common(trial, cfg, skip_lr=True)
        # LR must not have been touched by _suggest_common
        assert cfg["training"]["learning_rate"] == original_lr
        # Everything else should still be set
        assert cfg["training"]["weight_decay"] == 1e-3
        assert cfg["training"]["batch_size"] == 32


class TestSuggestModelSpecific:
    def test_inceptiontime(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "depth": 6,
                "n_filters": 32,
                "kernel_size": 41,
                "use_bottleneck": True,
                "bottleneck_size": 32,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "inceptiontime")
        assert cfg["model"]["depth"] == 6
        assert cfg["model"]["n_filters"] == 32
        assert cfg["model"]["kernel_size"] == 41
        assert cfg["model"]["use_bottleneck"] is True

    def test_ast(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "model_size": "base384",
                "learning_rate": 2e-5,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "ast")
        assert cfg["model"]["model_size"] == "base384"
        assert cfg["model"]["imagenet_pretrain"] is True
        assert cfg["model"]["audioset_pretrain"] is True

    def test_audio_mamba(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "embed_dim": 128,
                "depth": 4,
                "d_state": 16,
                "patch_size": 16,
            }
        )
        cfg = _base_cfg_dict()
        _suggest_model_specific(trial, cfg, "audio_mamba")
        assert cfg["model"]["embed_dim"] == 128
        assert cfg["model"]["depth"] == 4

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

    def test_persists_to_sqlite(self, tmp_path, monkeypatch) -> None:
        """Test that run_sweep creates a SQLite study database."""
        monkeypatch.setattr(
            "bioaed.hpo.optuna_sweep.objective",
            lambda trial, cfg: 0.5,
        )
        cfg = OmegaConf.create(_base_cfg_dict())
        study = run_sweep(
            cfg,
            n_trials=2,
            n_startup_trials=1,
            output_dir=tmp_path,
            study_name="test_persist",
        )
        assert (tmp_path / "study.db").exists()
        assert (tmp_path / "best_params.json").exists()
        assert len(study.trials) == 2

    def test_resumes_partial_study(self, tmp_path, monkeypatch) -> None:
        """Test that run_sweep resumes from an existing SQLite study."""
        call_count = 0

        def counting_objective(trial, cfg):
            nonlocal call_count
            call_count += 1
            return 0.5

        monkeypatch.setattr(
            "bioaed.hpo.optuna_sweep.objective",
            counting_objective,
        )
        cfg = OmegaConf.create(_base_cfg_dict())

        # First run: 2 trials
        run_sweep(
            cfg,
            n_trials=5,
            n_startup_trials=1,
            output_dir=tmp_path,
            study_name="test_resume",
        )
        # Simulate interruption: only 2 ran because we'll override n_trials
        # Actually, let's do it properly: run 2, then run 5 total
        call_count = 0
        study1 = run_sweep(
            cfg,
            n_trials=2,
            n_startup_trials=1,
            output_dir=tmp_path / "resume",
            study_name="test_resume2",
        )
        assert len(study1.trials) == 2

        # Second run: request 5 total — should only run 3 more
        call_count = 0
        study2 = run_sweep(
            cfg,
            n_trials=5,
            n_startup_trials=1,
            output_dir=tmp_path / "resume",
            study_name="test_resume2",
        )
        assert len(study2.trials) == 5
        assert call_count == 3  # Only 3 new trials executed

    def test_skips_when_all_done(self, tmp_path, monkeypatch) -> None:
        """Test that run_sweep skips optimization when all trials are done."""
        call_count = 0

        def counting_objective(trial, cfg):
            nonlocal call_count
            call_count += 1
            return 0.5

        monkeypatch.setattr(
            "bioaed.hpo.optuna_sweep.objective",
            counting_objective,
        )
        cfg = OmegaConf.create(_base_cfg_dict())

        # Run 3 trials
        run_sweep(
            cfg,
            n_trials=3,
            n_startup_trials=1,
            output_dir=tmp_path,
            study_name="test_skip",
        )
        assert call_count == 3

        # Re-run requesting same 3 — should run 0 new trials
        call_count = 0
        study = run_sweep(
            cfg,
            n_trials=3,
            n_startup_trials=1,
            output_dir=tmp_path,
            study_name="test_skip",
        )
        assert call_count == 0
        assert len(study.trials) == 3
