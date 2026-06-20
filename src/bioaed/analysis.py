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

    Scans ``outputs/ablation/{model}/{dataset}/seed_{N}/results.json``.

    Returns:
        DataFrame with columns: model, dataset, seed,
        and all metric columns from test_metrics.
    """
    rows = []

    for results_file in sorted(root.rglob("results.json")):
        with open(results_file) as f:
            data = json.load(f)

        # Extract config identifiers from path structure
        # .../model/dataset/seed_N/results.json
        parts = results_file.relative_to(root).parts
        if len(parts) < 3:
            logger.debug(f"Skipping unexpected path depth: {results_file}")
            continue
        model, dataset, seed_dir = parts[0], parts[1], parts[2]

        try:
            seed = int(seed_dir.split("_")[1])
        except (IndexError, ValueError):
            logger.debug(f"Skipping non-seed directory: {results_file}")
            continue

        test_metrics = data.get("test_metrics", {})
        row = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            **test_metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} ablation results from {root}")
    return df


def aggregate_results(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Aggregate per-config: mean, std, 95% CI via bootstrap.

    Groups by (model, dataset) and computes statistics across seeds.

    Args:
        df: Raw ablation results DataFrame.
        metric: Column name to aggregate.

    Returns:
        Summary DataFrame with mean, std, ci_lower, ci_upper per config.
    """
    group_cols = ["model", "dataset"]

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
                **dict(zip(group_cols, keys, strict=False)),
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


def build_comparison_matrix(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Build a model comparison matrix suitable for autorank.

    Each row = one seed, each column = one configuration.
    Required format for Friedman / Wilcoxon-Holm tests.

    Args:
        df: Raw ablation results DataFrame.
        metric: Column to use as the performance measure.

    Returns:
        Wide-format DataFrame with configs as columns and seeds as rows.
    """
    df = df.copy()
    df["config"] = df["model"] + "_" + df["dataset"]

    wide = df.pivot_table(index="seed", columns="config", values=metric)
    return wide


def build_model_comparison_matrix(df: pd.DataFrame, metric: str = "mAP") -> pd.DataFrame:
    """Build a model comparison matrix following Demšar (2006) for multi-dataset benchmarks.

    Each row is one (dataset, seed) block; each column is one model.  With 2 datasets
    and 5 seeds this produces a 10 × 3 matrix.  This formulation directly tests whether
    model differences are consistent across all observed dataset × seed combinations,
    which is the primary research question of the benchmark.

    Args:
        df: Raw ablation results DataFrame with columns model, dataset, seed, {metric}.
        metric: Column to use as the performance measure.

    Returns:
        Wide-format DataFrame with models as columns and (dataset, seed) blocks as rows.
    """
    _paper_names = {
        "audiomamba": "MambaSpec",
        "audiomamba_pretrained": "SSAMBA",
        "audiospectrogramtransformer": "AST",
    }
    df = df.copy()
    df["block"] = df["dataset"] + "_seed" + df["seed"].astype(str)
    wide = df.pivot_table(index="block", columns="model", values=metric)
    wide.columns = [_paper_names.get(c, c) for c in wide.columns]
    return wide


if __name__ == "__main__":
    results = load_ablation_results()
    if not results.empty:
        summary = aggregate_results(results, metric="mAP")
        print(summary.to_string(index=False))
