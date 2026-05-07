"""Tests for Hydra config loading and Pydantic validation."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from bioaed.utils.config_schemas import (
    AugmentationConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig validation."""

    def test_valid_aswine_config(self) -> None:
        """Valid aSwine config should pass validation."""
        cfg = DatasetConfig(
            name="aswine",
            root_dir="data/aswine",
            sample_rate=16000,
            segment_duration=1.0,
            hop_length=160,
            win_length=400,
            num_classes=7,
            meta_variant="1s_pruned",
        )
        assert cfg.name == "aswine"
        assert cfg.num_classes == 7

    def test_valid_anuraset_config(self) -> None:
        """Valid AnuraSet config should pass validation."""
        cfg = DatasetConfig(
            name="anuraset",
            root_dir="data/anuraset",
            sample_rate=22050,
            segment_duration=3.0,
            hop_length=220,
            win_length=550,
            num_classes=42,
        )
        assert cfg.name == "anuraset"
        assert cfg.num_classes == 42

    def test_invalid_dataset_name(self) -> None:
        """Invalid dataset name should raise validation error."""
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="invalid",  # type: ignore[arg-type]
                root_dir="data/invalid",
                sample_rate=16000,
                segment_duration=1.0,
                hop_length=160,
                win_length=400,
                num_classes=7,
            )

    def test_invalid_sample_rate(self) -> None:
        """Negative sample rate should raise validation error."""
        with pytest.raises(ValidationError):
            DatasetConfig(
                name="aswine",
                root_dir="data/aswine",
                sample_rate=-1,
                segment_duration=1.0,
                hop_length=160,
                win_length=400,
                num_classes=7,
            )


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_defaults(self) -> None:
        """Default training config should have sensible values."""
        cfg = TrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.learning_rate == 1e-3
        assert cfg.precision == "bf16-mixed"

    def test_invalid_precision(self) -> None:
        """Invalid precision string should raise validation error."""
        with pytest.raises(ValidationError):
            TrainingConfig(precision="fp8")  # type: ignore[arg-type]


class TestExperimentConfig:
    """Tests for full experiment config validation."""

    def test_from_omegaconf(self, aswine_cfg: OmegaConf) -> None:
        """Full config loaded from OmegaConf should validate."""
        cfg_dict = OmegaConf.to_container(aswine_cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        config = ExperimentConfig(**cfg_dict)
        assert config.experiment_name == "test"
        assert config.dataset.name == "aswine"


class TestAugmentationConfig:
    """Tests for AugmentationConfig validation."""

    def test_defaults(self) -> None:
        cfg = AugmentationConfig()
        assert cfg.enabled is True
        assert cfg.freq_mask_param == 8
        assert cfg.time_mask_param == 20

    def test_custom_values(self) -> None:
        cfg = AugmentationConfig(
            enabled=False,
            freq_mask_param=16,
            time_mask_param=40,
            n_freq_masks=3,
            n_time_masks=3,
        )
        assert cfg.enabled is False
        assert cfg.n_freq_masks == 3

    def test_invalid_freq_mask(self) -> None:
        with pytest.raises(ValidationError):
            AugmentationConfig(freq_mask_param=-1)


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_allows_extra_fields(self) -> None:
        cfg = ModelConfig(
            _target_="bioaed.models.inceptiontime.InceptionTime", num_classes=7, depth=3
        )
        assert cfg.num_classes == 7


class TestTrainingConfigExtended:
    """Extended TrainingConfig tests."""

    def test_compile_flag(self) -> None:
        cfg = TrainingConfig(compile=True)
        assert cfg.compile is True

    def test_scheduler_options(self) -> None:
        for sched in ["cosine", "step", "none"]:
            cfg = TrainingConfig(scheduler=sched)
            assert cfg.scheduler == sched

    def test_invalid_scheduler(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(scheduler="cyclic")  # type: ignore[arg-type]
