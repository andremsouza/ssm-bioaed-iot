"""Classification metrics for multi-label bioacoustic audio event detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute multi-label classification metrics.

    Args:
        logits: Raw model outputs of shape ``(N, num_classes)``.
        labels: Ground-truth binary labels of shape ``(N, num_classes)``.

    Returns:
        Dictionary with ``roc_auc_weighted``, ``mAP``, and per-class AUC.
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))

    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    metrics: dict[str, Any] = {}

    # Weighted ROC-AUC
    try:
        metrics["roc_auc_weighted"] = float(
            roc_auc_score(labels_np, probs, average="weighted", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_weighted"] = 0.0

    # Mean Average Precision (mAP)
    try:
        metrics["mAP"] = float(average_precision_score(labels_np, probs, average="weighted"))
    except ValueError:
        metrics["mAP"] = 0.0

    # Per-class AUC
    num_classes = labels_np.shape[1]
    per_class_auc: list[float] = []
    for i in range(num_classes):
        try:
            auc = float(roc_auc_score(labels_np[:, i], probs[:, i]))
        except ValueError:
            auc = 0.0
        per_class_auc.append(auc)
        metrics[f"auc_class_{i}"] = auc

    metrics["auc_macro"] = float(np.mean(per_class_auc))

    return metrics
