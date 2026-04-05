"""Publication-quality visualizations for ablation results.

Generates:
    - Critical difference (CD) diagrams via ``autorank``.
    - Ablation heatmaps showing DCQG delta per model × dataset.
    - LaTeX-formatted results tables.
"""

from __future__ import annotations

from pathlib import Path

import autorank
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from bioaed.evaluation.statistical_tests import StatisticalResult

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_cd_diagram(
    result: StatisticalResult,
    output_path: Path | None = None,
) -> None:
    """Plot Critical Difference diagram using autorank.

    Args:
        result: Output from :func:`~bioaed.evaluation.statistical_tests.run_statistical_comparison`.
        output_path: Where to save. Defaults to ``reports/figures/cd_{metric}.pdf``.
    """
    _ensure_dirs()
    if result.autorank_result is None:
        logger.warning("No autorank result available; skipping CD diagram.")
        return

    if output_path is None:
        output_path = FIGURES_DIR / f"cd_{result.metric}.pdf"

    fig, ax = plt.subplots(figsize=(10, 4))
    autorank.plot_stats(result.autorank_result, ax=ax, allow_insignificant=True)
    ax.set_title(f"Critical Difference — {result.metric}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CD diagram saved to {output_path}")


def plot_ablation_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "mAP",
    output_path: Path | None = None,
) -> None:
    """Plot DCQG on-vs-off delta heatmap (model × dataset).

    Args:
        summary_df: Aggregated results from :func:`~bioaed.analysis.aggregate_results`.
        metric: Metric name.
        output_path: Where to save. Defaults to ``reports/figures/heatmap_{metric}.pdf``.
    """
    _ensure_dirs()
    mean_col = f"{metric}_mean"

    # Pivot to get DCQG on/off per model × dataset
    on = summary_df[summary_df["dcqg"] == "dcqg_on"].set_index(["model", "dataset"])[mean_col]
    off = summary_df[summary_df["dcqg"] == "dcqg_off"].set_index(["model", "dataset"])[mean_col]
    delta = (on - off).unstack("dataset")

    if delta.empty:
        logger.warning("No paired DCQG on/off data; skipping heatmap.")
        return

    if output_path is None:
        output_path = FIGURES_DIR / f"heatmap_{metric}.pdf"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        delta,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"DCQG Delta ({metric}): on − off")
    ax.set_ylabel("Model")
    ax.set_xlabel("Dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved to {output_path}")


def plot_metric_boxplots(
    raw_df: pd.DataFrame,
    metric: str = "mAP",
    output_path: Path | None = None,
) -> None:
    """Boxplots of metric distributions per model × DCQG state.

    Args:
        raw_df: Raw ablation results DataFrame.
        metric: Column to plot.
        output_path: Where to save.
    """
    _ensure_dirs()
    if metric not in raw_df.columns:
        logger.warning(f"Metric '{metric}' not in results; skipping boxplots.")
        return

    if output_path is None:
        output_path = FIGURES_DIR / f"boxplot_{metric}.pdf"

    df = raw_df.copy()
    df["config"] = df["model"] + " / " + df["dcqg"]

    fig, axes = plt.subplots(
        1, df["dataset"].nunique(), figsize=(7 * df["dataset"].nunique(), 5), squeeze=False
    )
    for i, (ds_name, ds_group) in enumerate(df.groupby("dataset")):
        ax = axes[0, i]
        sns.boxplot(data=ds_group, x="model", y=metric, hue="dcqg", ax=ax, palette="Set2")
        ax.set_title(f"{ds_name}")
        ax.set_xlabel("")
    fig.suptitle(f"{metric} Distribution by Model and DCQG State", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplots saved to {output_path}")


def generate_latex_table(
    summary_df: pd.DataFrame,
    metric: str = "mAP",
    output_path: Path | None = None,
) -> str:
    """Generate a LaTeX table from aggregated results.

    Args:
        summary_df: Aggregated results.
        metric: Base metric name.
        output_path: Where to save .tex. Defaults to ``reports/table_{metric}.tex``.

    Returns:
        LaTeX table string.
    """
    _ensure_dirs()
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    df = summary_df.copy()
    df["result"] = df.apply(lambda r: f"{r[mean_col]:.4f} $\\pm$ {r[std_col]:.4f}", axis=1)
    pivot = df.pivot_table(
        index="model", columns=["dataset", "dcqg"], values="result", aggfunc="first"
    )

    latex = pivot.to_latex(
        escape=False, multicolumn=True, caption=f"{metric} Results", label=f"tab:{metric}"
    )

    if output_path is None:
        output_path = REPORTS_DIR / f"table_{metric}.tex"
    output_path.write_text(latex)
    logger.info(f"LaTeX table saved to {output_path}")
    return latex
