"""Result aggregation and analysis for ablation studies.

Loads structured JSON results from ablation runs, aggregates per-config
statistics (mean, std, 95% CI), and generates comparison DataFrames
for statistical testing and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ABLATION_ROOT = Path("outputs/ablation")


def load_ablation_results(root: Path = ABLATION_ROOT) -> pd.DataFrame:
    """Load all ablation result JSONs into a single DataFrame.

    Scans ``outputs/ablation/{model}/{dataset}/{dcqg}/{seed}/results.json``
    and optionally ``cv_results.json`` for fold-level results.

    Returns:
        DataFrame with columns: model, dataset, dcqg, seed, fold,
        and all metric columns from test_metrics.
    """
    rows = []

    for results_file in sorted(root.rglob("results.json")):
        # Skip cv-aggregated files (we read individual folds)
        if results_file.parent.name.startswith("fold_"):
            fold_idx = int(results_file.parent.name.split("_")[1])
        else:
            fold_idx = None

        with open(results_file) as f:
            data = json.load(f)

        # Extract config identifiers from path structure
        parts = results_file.relative_to(root).parts
        if fold_idx is not None:
            # .../model/dataset/dcqg/seed_N/fold_K/results.json
            model, dataset, dcqg, seed_dir = parts[0], parts[1], parts[2], parts[3]
        else:
            # .../model/dataset/dcqg/seed_N/results.json
            model, dataset, dcqg, seed_dir = parts[0], parts[1], parts[2], parts[3]

        seed = int(seed_dir.split("_")[1])

        test_metrics = data.get("test_metrics", {})
        row = {
            "model": model,
            "dataset": dataset,
            "dcqg": dcqg,
            "seed": seed,
            "fold": fold_idx,
            **test_metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} ablation results from {root}")
    return df


def aggregate_results(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Aggregate per-config: mean, std, 95% CI via bootstrap.

    Groups by (model, dataset, dcqg) and computes statistics across seeds/folds.

    Args:
        df: Raw ablation results DataFrame.
        metric: Column name to aggregate.

    Returns:
        Summary DataFrame with mean, std, ci_lower, ci_upper per config.
    """
    group_cols = ["model", "dataset", "dcqg"]

    def _bootstrap_ci(
        values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
    ) -> tuple[float, float]:
        rng = np.random.default_rng(42)
        boot_means = np.array(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        )
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return lo, hi

    records = []
    for keys, group in df.groupby(group_cols, sort=False):
        vals = group[metric].dropna().values
        if len(vals) == 0:
            continue
        ci_lo, ci_hi = _bootstrap_ci(vals)
        records.append(
            {
                **dict(zip(group_cols, keys)),
                f"{metric}_mean": float(vals.mean()),
                f"{metric}_std": float(vals.std()),
                f"{metric}_ci_lower": ci_lo,
                f"{metric}_ci_upper": ci_hi,
                "n_runs": len(vals),
            }
        )

    summary = pd.DataFrame(records)
    logger.info(f"Aggregated {len(summary)} configs for metric '{metric}'")
    return summary


def compute_dcqg_delta(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Compute DCQG on-vs-off delta per model × dataset.

    Args:
        df: Aggregated summary DataFrame (from :func:`aggregate_results`).
        metric: Base metric name.

    Returns:
        DataFrame with delta_mean and columns for each model × dataset.
    """
    mean_col = f"{metric}_mean"
    pivot = df.pivot_table(
        index=["model", "dataset"], columns="dcqg", values=mean_col
    ).reset_index()

    if "dcqg_on" in pivot.columns and "dcqg_off" in pivot.columns:
        pivot["delta"] = pivot["dcqg_on"] - pivot["dcqg_off"]

    return pivot


def build_comparison_matrix(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Build a (model × dcqg) comparison matrix suitable for autorank.

    Each row = one seed/fold, each column = one configuration.
    Required format for Friedman / Wilcoxon-Holm tests.

    Args:
        df: Raw ablation results DataFrame.
        metric: Column to use as the performance measure.

    Returns:
        Wide-format DataFrame with configs as columns and seeds as rows.
    """
    df = df.copy()
    df["config"] = df["model"] + "_" + df["dataset"] + "_" + df["dcqg"]

    # Pivot: rows = seed (+ optional fold), columns = config
    id_cols = ["seed"]
    if "fold" in df.columns and df["fold"].notna().any():
        id_cols.append("fold")

    wide = df.pivot_table(index=id_cols, columns="config", values=metric)
    return wide


if __name__ == "__main__":
    results = load_ablation_results()
    if not results.empty:
        summary = aggregate_results(results, metric="mAP")
        print(summary.to_string(index=False))

        delta = compute_dcqg_delta(summary, metric="mAP")
        print("\nDCQG Delta:")
        print(delta.to_string(index=False))
