"""Integration tests for dataset adapters and data module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from bioaed.data.anuraset import AnuraSetDataset
from bioaed.data.aswine import ASwineDataset, ASWINE_LABEL_COLUMNS
from bioaed.data.datamodule import BioacousticDataModule
from bioaed.features.quality_gate import QualityGate

ASWINE_ROOT = Path("data/aswine")
ANURASET_ROOT = Path("data/anuraset")

needs_aswine = pytest.mark.skipif(
    not (ASWINE_ROOT / "meta" / "1s_pruned" / "aswine_1s_pruned_train.csv").exists(),
    reason="aSwine data not available",
)
needs_anuraset = pytest.mark.skipif(
    not (ANURASET_ROOT / "metadata.csv").exists(),
    reason="AnuraSet data not available",
)


class TestASwineDataset:
    """Integration tests for the ASwineDataset adapter."""

    @needs_aswine
    def test_load_metadata(self) -> None:
        """Metadata should load with correct label count."""
        ds = ASwineDataset(
            root_dir=ASWINE_ROOT,
            meta_variant="1s_pruned",
            split="train",
            sample_rate=16000,
            segment_duration=1.0,
            num_classes=7,
            hop_length=160,
            win_length=400,
        )
        assert len(ds) > 0
        row = ds.metadata[0]
        assert len(row["labels"]) == 7
        assert all(lbl in (0, 1) for lbl in row["labels"])

    @needs_aswine
    def test_getitem_shapes(self) -> None:
        """A single sample should return (spectrogram, labels, weight) with correct shapes."""
        ds = ASwineDataset(
            root_dir=ASWINE_ROOT,
            meta_variant="1s_pruned",
            split="train",
            sample_rate=16000,
            segment_duration=1.0,
            num_classes=7,
            n_mels=64,
            hop_length=160,
            win_length=400,
        )
        spec, labels, weight = ds[0]
        assert spec.shape[0] == 64  # n_mels
        assert labels.shape == (7,)
        assert weight.shape == ()
        assert weight.item() == pytest.approx(1.0)  # no quality gate

    @needs_aswine
    def test_getitem_with_quality_gate(self) -> None:
        """Quality gate should produce a weight in [0, 1]."""
        gate = QualityGate(snr_threshold=5.0, spectral_flatness_threshold=0.5)
        ds = ASwineDataset(
            root_dir=ASWINE_ROOT,
            meta_variant="1s_pruned",
            split="train",
            sample_rate=16000,
            segment_duration=1.0,
            num_classes=7,
            hop_length=160,
            win_length=400,
            quality_gate=gate,
        )
        _, _, weight = ds[0]
        assert 0.0 <= weight.item() <= 1.0


class TestAnuraSetDataset:
    """Integration tests for the AnuraSetDataset adapter."""

    @needs_anuraset
    def test_load_metadata(self) -> None:
        """Metadata should load with exactly 42 species labels."""
        ds = AnuraSetDataset(
            root_dir=ANURASET_ROOT,
            split="test",
            sample_rate=22050,
            segment_duration=3.0,
            num_classes=42,
            hop_length=220,
            win_length=550,
        )
        assert len(ds) > 0
        row = ds.metadata[0]
        assert len(row["labels"]) == 42
        assert all(lbl in (0, 1) for lbl in row["labels"])

    @needs_anuraset
    def test_getitem_shapes(self) -> None:
        """A single sample should have the correct spectrogram and label shapes."""
        ds = AnuraSetDataset(
            root_dir=ANURASET_ROOT,
            split="test",
            sample_rate=22050,
            segment_duration=3.0,
            num_classes=42,
            n_mels=64,
            hop_length=220,
            win_length=550,
        )
        spec, labels, weight = ds[0]
        assert spec.shape[0] == 64  # n_mels
        assert labels.shape == (42,)
        assert weight.item() == pytest.approx(1.0)

    @needs_anuraset
    def test_audio_path_resolution(self) -> None:
        """Audio path should point to an existing file with site subdirectory."""
        ds = AnuraSetDataset(
            root_dir=ANURASET_ROOT,
            split="test",
            sample_rate=22050,
            segment_duration=3.0,
            num_classes=42,
            hop_length=220,
            win_length=550,
        )
        audio_path, offset = ds._get_audio_path_and_offset(0)
        assert audio_path.exists(), f"Audio file not found: {audio_path}"
        assert offset == 0.0


class TestBioacousticDataModule:
    """Integration tests for the data module factory."""

    @needs_aswine
    def test_aswine_setup(self) -> None:
        """DataModule should create train/val/test splits for aSwine."""
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "dataset": {
                    "name": "aswine",
                    "root_dir": str(ASWINE_ROOT),
                    "meta_variant": "1s_pruned",
                    "sample_rate": 16000,
                    "segment_duration": 1.0,
                    "n_mels": 64,
                    "hop_length": 160,
                    "win_length": 400,
                    "num_classes": 7,
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "quality_gate": {
                    "enabled": False,
                    "snr_threshold": 0.0,
                    "spectral_flatness_threshold": 0.0,
                    "weighting_strategy": "soft",
                },
            }
        )
        dm = BioacousticDataModule(cfg)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

    @needs_aswine
    def test_aswine_dataloader_batch(self) -> None:
        """A single batch from the train dataloader should have correct shapes."""
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "dataset": {
                    "name": "aswine",
                    "root_dir": str(ASWINE_ROOT),
                    "meta_variant": "1s_pruned",
                    "sample_rate": 16000,
                    "segment_duration": 1.0,
                    "n_mels": 64,
                    "hop_length": 160,
                    "win_length": 400,
                    "num_classes": 7,
                    "batch_size": 4,
                    "num_workers": 0,
                    "pin_memory": False,
                },
                "quality_gate": {
                    "enabled": True,
                    "snr_threshold": 0.0,
                    "spectral_flatness_threshold": 0.0,
                    "weighting_strategy": "soft",
                },
            }
        )
        dm = BioacousticDataModule(cfg)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        spec, labels, weights = batch
        assert spec.shape[0] == 4  # batch size
        assert spec.shape[1] == 64  # n_mels
        assert labels.shape == (4, 7)
        assert weights.shape == (4,)
