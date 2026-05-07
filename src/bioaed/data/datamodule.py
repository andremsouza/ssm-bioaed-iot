"""Bioacoustic data module wrapping dataset creation and DataLoader setup."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from bioaed.data.anuraset import AnuraSetDataset
from bioaed.data.aswine import ASwineDataset
from bioaed.features.transforms import SpecAugment, ZNormalize

# Dataset factory registry
_DATASET_REGISTRY: dict[str, type] = {
    "aswine": ASwineDataset,
    "anuraset": AnuraSetDataset,
}


def _seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility (PyTorch recommendation)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


class BioacousticDataModule:
    """Factory-based data module for bioacoustic datasets.

    Creates train/val/test datasets and their corresponding DataLoaders
    based on the Hydra dataset configuration.

    Args:
        cfg: OmegaConf DictConfig with ``dataset`` group.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None
        self.znormalize: ZNormalize | None = None

    def setup(self) -> None:
        """Instantiate train, val, and test datasets with 85/15 stratified split."""
        full_train = self._create_raw_datasets()
        train_indices, val_indices = self._stratified_split(full_train, val_ratio=0.15)
        self._finalize_split(full_train, train_indices, val_indices)

    def setup_fold(self, fold_idx: int = 0, n_folds: int = 5) -> None:
        """Setup for k-fold cross-validation (specific fold)."""
        full_train = self._create_raw_datasets()
        train_indices, val_indices = self._kfold_split(full_train, fold_idx, n_folds)
        self._finalize_split(full_train, train_indices, val_indices)

    def _create_raw_datasets(self) -> Any:
        """Create raw train and test dataset instances."""
        dataset_name = self.cfg.dataset.name
        dataset_cls = _DATASET_REGISTRY.get(dataset_name)
        if dataset_cls is None:
            msg = f"Unknown dataset: {dataset_name}. Available: {list(_DATASET_REGISTRY)}"
            raise ValueError(msg)

        # Common kwargs passed to the dataset constructor
        common_kwargs: dict[str, Any] = {
            "root_dir": self.cfg.dataset.root_dir,
            "sample_rate": self.cfg.dataset.sample_rate,
            "segment_duration": self.cfg.dataset.segment_duration,
            "n_mels": self.cfg.dataset.n_mels,
            "hop_length": self.cfg.dataset.hop_length,
            "win_length": self.cfg.dataset.win_length,
            "num_classes": self.cfg.dataset.num_classes,
        }

        # Dataset-specific kwargs
        if dataset_name == "aswine":
            common_kwargs["meta_variant"] = self.cfg.dataset.meta_variant

        full_train = dataset_cls(split="train", **common_kwargs)
        self.test_dataset = dataset_cls(split="test", **common_kwargs)

        # Configure SpecAugment for training data only
        aug_cfg = getattr(self.cfg, "augmentation", None)
        if aug_cfg is not None and getattr(aug_cfg, "enabled", False):
            full_train.augment = SpecAugment(
                freq_mask_param=getattr(aug_cfg, "freq_mask_param", 8),
                time_mask_param=getattr(aug_cfg, "time_mask_param", 20),
                n_freq_masks=getattr(aug_cfg, "n_freq_masks", 2),
                n_time_masks=getattr(aug_cfg, "n_time_masks", 2),
            )

        return full_train

    def _finalize_split(
        self,
        full_train: Any,
        train_indices: list[int],
        val_indices: list[int],
    ) -> None:
        """Apply split, normalization, and log distributions."""
        self.train_dataset = Subset(full_train, train_indices)
        self.val_dataset = Subset(full_train, val_indices)

        # Compute and apply Z-normalization if enabled
        if getattr(self.cfg.dataset, "normalize", False):
            znorm = self._compute_normalization_stats(full_train, train_indices)
            full_train.normalize = znorm
            self.test_dataset.normalize = znorm  # type: ignore[union-attr]
            self.znormalize = znorm

        # Log class distributions
        self._log_class_distributions(full_train, train_indices, val_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stratified_split(
        self, dataset: Any, val_ratio: float = 0.15
    ) -> tuple[list[int], list[int]]:
        """Stratified train/val split preserving multi-label distribution."""
        labels = np.array([row["labels"] for row in dataset.metadata])
        label_keys = ["".join(str(int(x)) for x in row) for row in labels]

        # Group rare label combinations (< 2 occurrences) to avoid stratification failure
        key_counts = Counter(label_keys)
        label_keys_safe = [k if key_counts[k] >= 2 else "_rare_" for k in label_keys]

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=self.cfg.seed
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(label_keys_safe)), label_keys_safe))
        return train_idx.tolist(), val_idx.tolist()

    def _kfold_split(
        self, dataset: Any, fold_idx: int, n_folds: int
    ) -> tuple[list[int], list[int]]:
        """K-fold stratified split for cross-validation."""
        labels = np.array([row["labels"] for row in dataset.metadata])
        label_keys = ["".join(str(int(x)) for x in row) for row in labels]

        key_counts = Counter(label_keys)
        label_keys_safe = [k if key_counts[k] >= n_folds else "_rare_" for k in label_keys]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.cfg.seed)
        splits = list(skf.split(np.zeros(len(label_keys_safe)), label_keys_safe))
        train_idx, val_idx = splits[fold_idx]
        return train_idx.tolist(), val_idx.tolist()

    def _compute_normalization_stats(
        self, dataset: Any, train_indices: list[int], max_samples: int = 1000
    ) -> ZNormalize:
        """Compute per-channel mean/std from a sample of training spectrograms."""
        rng = np.random.default_rng(self.cfg.seed)
        n = min(max_samples, len(train_indices))
        idx_sample = rng.choice(train_indices, size=n, replace=False)

        channel_sum = torch.zeros(dataset.n_mels)
        channel_sq_sum = torch.zeros(dataset.n_mels)
        frame_count = 0

        for idx in idx_sample:
            spec, _ = dataset[int(idx)]
            channel_sum += spec.sum(dim=1)
            channel_sq_sum += (spec**2).sum(dim=1)
            frame_count += spec.shape[1]

        mean = (channel_sum / frame_count).unsqueeze(1)
        var = channel_sq_sum / frame_count - mean.squeeze(1).pow(2)
        std = var.clamp(min=1e-9).sqrt().unsqueeze(1)

        logger.info(f"ZNormalize stats computed from {n} training samples ({frame_count} frames)")
        return ZNormalize.from_dataset_stats(mean=mean, std=std)

    def _log_class_distributions(
        self,
        dataset: Any,
        train_indices: list[int],
        val_indices: list[int],
    ) -> None:
        """Log per-class label frequencies for train/val/test splits."""
        labels_all = np.array([row["labels"] for row in dataset.metadata])
        train_dist = labels_all[train_indices].mean(axis=0)
        val_dist = labels_all[val_indices].mean(axis=0)

        test_labels = np.array(
            [row["labels"] for row in self.test_dataset.metadata]  # type: ignore[union-attr]
        )
        test_dist = test_labels.mean(axis=0)

        logger.info(f"Train class frequencies: {train_dist.round(4).tolist()}")
        logger.info(f"Val   class frequencies: {val_dist.round(4).tolist()}")
        logger.info(f"Test  class frequencies: {test_dist.round(4).tolist()}")
        logger.info(
            f"Split sizes — Train: {len(train_indices)}, Val: {len(val_indices)}, "
            f"Test: {len(self.test_dataset)}"  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader[Any]:
        """Create the training DataLoader."""
        assert self.train_dataset is not None, "Call setup() first"
        nw = self.cfg.dataset.num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=self.cfg.dataset.pin_memory,
            persistent_workers=nw > 0,
            drop_last=True,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create the validation DataLoader."""
        assert self.val_dataset is not None, "Call setup() first"
        nw = self.cfg.dataset.num_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=self.cfg.dataset.pin_memory,
            persistent_workers=nw > 0,
            worker_init_fn=_seed_worker,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create the test DataLoader."""
        assert self.test_dataset is not None, "Call setup() first"
        nw = self.cfg.dataset.num_workers
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=self.cfg.dataset.pin_memory,
            persistent_workers=nw > 0,
            worker_init_fn=_seed_worker,
        )
