"""Tests for classification metrics."""

from __future__ import annotations

import numpy as np
import torch

from bioaed.evaluation.metrics import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self) -> None:
        labels = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        # Very high logits for positive, very low for negative
        logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
        metrics = compute_metrics(logits, labels)
        assert "mAP" in metrics
        assert "roc_auc_weighted" in metrics
        assert metrics["mAP"] > 0.9
        assert metrics["roc_auc_weighted"] > 0.9

    def test_random_predictions(self) -> None:
        torch.manual_seed(42)
        labels = torch.randint(0, 2, (50, 5)).float()
        logits = torch.randn(50, 5)
        metrics = compute_metrics(logits, labels)
        assert "mAP" in metrics
        assert "auc_macro" in metrics
        assert 0.0 <= metrics["mAP"] <= 1.0

    def test_numpy_input(self) -> None:
        labels = np.array([[1, 0], [0, 1]], dtype=np.float32)
        logits = np.array([[5.0, -5.0], [-5.0, 5.0]])
        metrics = compute_metrics(logits, labels)
        assert metrics["mAP"] > 0.9

    def test_per_class_auc_present(self) -> None:
        labels = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        logits = torch.tensor([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])
        metrics = compute_metrics(logits, labels)
        assert "auc_class_0" in metrics
        assert "auc_class_1" in metrics
        assert "auc_class_2" in metrics

    def test_single_class_all_same_label(self) -> None:
        """When one class has all zeros, AUC is undefined (nan)."""
        import math

        labels = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)
        logits = torch.randn(2, 2)
        metrics = compute_metrics(logits, labels)
        # Class 1 has all zeros — AUC is undefined
        assert math.isnan(metrics["auc_class_1"])
