"""Tests for Hydra entry-point modules (import coverage + helper logic)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

_CONFIGS_DIR = str(Path(__file__).parent.parent / "configs")


class TestEvaluateImport:
    def test_module_imports(self) -> None:
        import bioaed.evaluate  # noqa: F401


class TestTrainImport:
    def test_module_imports(self) -> None:
        import bioaed.train  # noqa: F401


class TestSweepImport:
    def test_module_imports(self) -> None:
        import bioaed.sweep  # noqa: F401


def _compose_cfg(overrides: list[str] | None = None):
    """Helper: compose a valid DictConfig using the project's Hydra configs."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONFIGS_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides or [])


class TestTrainMainLogic:
    """Cover train.py main() body with mocked data and trainer."""

    def test_main_runs_with_mocked_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _compose_cfg()

        with (
            patch("bioaed.train.BioacousticDataModule") as mock_dm_cls,
            patch("bioaed.train.FabricTrainer") as mock_trainer_cls,
            patch("bioaed.train.Fabric") as mock_fabric_cls,
        ):
            mock_dm = mock_dm_cls.return_value
            mock_dm.train_dataloader.return_value = []
            mock_dm.val_dataloader.return_value = []

            mock_trainer = mock_trainer_cls.return_value
            mock_trainer.fit.return_value = {"val_loss": 0.4, "mAP": 0.6}

            mock_fabric_cls.return_value = MagicMock()

            from bioaed.train import main

            main(cfg)

        mock_dm.setup.assert_called_once()
        mock_trainer.fit.assert_called_once()


class TestEvaluateMainLogic:
    """Cover evaluate.py main() body with mocked data and model."""

    def test_main_runs_with_mocked_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _compose_cfg()

        # Fake batch: (spectrogram, labels, weights)
        fake_batch = (
            torch.randn(2, cfg.dataset.n_mels, 50),
            torch.zeros(2, cfg.dataset.num_classes),
            torch.ones(2),
        )

        with (
            patch("bioaed.evaluate.BioacousticDataModule") as mock_dm_cls,
            patch("bioaed.evaluate.build_model") as mock_build,
        ):
            from bioaed.models.inceptiontime import InceptionTime

            model = InceptionTime(
                num_classes=cfg.dataset.num_classes,
                in_channels=cfg.dataset.n_mels,
                depth=1,
                n_filters=16,
            )
            mock_build.return_value = model
            mock_dm_cls.return_value.test_dataloader.return_value = [fake_batch]

            from bioaed.evaluate import main

            main(cfg)

        out_file = tmp_path / "outputs" / "evaluation_results.json"
        assert out_file.exists()


class TestSweepMainLogic:
    """Cover sweep.py main() body with a mocked HPO run."""

    def test_main_calls_run_sweep(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _compose_cfg()

        mock_study = MagicMock()
        mock_study.best_trial.number = 0
        mock_study.best_value = 0.35
        mock_study.best_params = {"learning_rate": 1e-3}

        with patch("bioaed.sweep.run_sweep", return_value=mock_study) as mock_sweep:
            from bioaed.sweep import main

            main(cfg)

        mock_sweep.assert_called_once()

