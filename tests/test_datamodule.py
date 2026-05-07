"""Integration tests for BioacousticDataModule using real aSwine data."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bioaed.data.datamodule import BioacousticDataModule, _seed_worker

DATA_DIR = Path("data/aswine")
_DATA_AVAILABLE = (DATA_DIR / "meta" / "1s_pruned" / "aswine_1s_pruned_train.csv").exists()


def _make_cfg(augmentation_enabled: bool = False, normalize: bool = False) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
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
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "normalize": normalize,
            },
            "augmentation": {
                "enabled": augmentation_enabled,
                "freq_mask_param": 8,
                "time_mask_param": 20,
                "n_freq_masks": 2,
                "n_time_masks": 2,
            },
        }
    )


class TestSeedWorker:
    def test_does_not_raise(self) -> None:
        _seed_worker(0)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="aSwine data not available")
class TestDataModuleSetup:
    def test_setup_creates_splits(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0

    def test_setup_fold(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup_fold(fold_idx=0, n_folds=3)
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_train_dataloader(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup()
        dl = dm.train_dataloader()
        assert isinstance(dl, DataLoader)

    def test_val_dataloader(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup()
        dl = dm.val_dataloader()
        assert isinstance(dl, DataLoader)

    def test_test_dataloader(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup()
        dl = dm.test_dataloader()
        assert isinstance(dl, DataLoader)

    def test_with_augmentation(self) -> None:
        dm = BioacousticDataModule(_make_cfg(augmentation_enabled=True))
        dm.setup()
        assert dm.train_dataset is not None

    def test_with_meta_variant(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        dm.setup()
        assert dm.train_dataset is not None

    def test_dataloader_not_setup_asserts(self) -> None:
        dm = BioacousticDataModule(_make_cfg())
        with pytest.raises(AssertionError):
            dm.train_dataloader()
        with pytest.raises(AssertionError):
            dm.val_dataloader()
        with pytest.raises(AssertionError):
            dm.test_dataloader()

    def test_with_normalize(self) -> None:
        """Setting normalize=True triggers _compute_normalization_stats."""
        dm = BioacousticDataModule(_make_cfg(normalize=True))
        dm.setup()
        assert dm.znormalize is not None
        assert dm.train_dataset is not None


class TestDataModuleErrors:
    """Unit tests for error paths that do not require real data."""

    def test_unknown_dataset_raises(self) -> None:
        """Requesting an unknown dataset name must raise ValueError."""
        cfg = OmegaConf.create(
            {
                "seed": 42,
                "dataset": {
                    "name": "nonexistent_dataset",
                    "root_dir": "data/nonexistent",
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
            }
        )
        dm = BioacousticDataModule(cfg)
        with pytest.raises(ValueError, match="Unknown dataset"):
            dm.setup()
