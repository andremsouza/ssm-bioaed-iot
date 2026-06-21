"""Publication-quality visualizations for ablation results.

Generates:
    - Critical difference (CD) diagrams via ``autorank``.
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

    # Let autorank compute its natural compact height for the number of models.
    # Using rc_context keeps the font-size change local to this call.
    with plt.rc_context({"font.size": 12}):
        ax = autorank.plot_stats(result.autorank_result, allow_insignificant=True, width=6)
    if ax is None:
        logger.warning("autorank returned no axis; skipping CD diagram.")
        return
    fig = ax.get_figure()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CD diagram saved to {output_path}")


def plot_metric_boxplots(
    raw_df: pd.DataFrame,
    metric: str = "mAP",
    output_path: Path | None = None,
) -> None:
    """Boxplots of metric distributions per model.

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
    n_datasets = df["dataset"].nunique()
    if n_datasets == 0:
        logger.warning(f"No rows to plot for metric '{metric}'; skipping boxplots.")
        return

    fig, axes = plt.subplots(
        1, n_datasets, figsize=(7 * n_datasets, 5), squeeze=False
    )
    for i, (ds_name, ds_group) in enumerate(df.groupby("dataset")):
        ax = axes[0, i]
        sns.boxplot(data=ds_group, x="model", y=metric, ax=ax, palette="Set2")
        ax.set_title(f"{ds_name}")
        ax.set_xlabel("")
    fig.suptitle(f"{metric} Distribution by Model", y=1.02)
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
    pivot = df.pivot_table(index="model", columns="dataset", values="result", aggfunc="first")

    latex = pivot.to_latex(
        escape=False, multicolumn=True, caption=f"{metric} Results", label=f"tab:{metric}"
    )

    if output_path is None:
        output_path = REPORTS_DIR / f"table_{metric}.tex"
    output_path.write_text(latex)
    logger.info(f"LaTeX table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# Benchmark-specific visualizations (3-config paper)
# ---------------------------------------------------------------------------

# Friendly display names for model keys (paper names)
_MODEL_DISPLAY = {
    "audiospectrogramtransformer": "AST",
    "audiomamba": "MambaSpec",
    "audiomamba_pretrained": "SSAMBA",
}


def _rename_model(name: str) -> str:
    return _MODEL_DISPLAY.get(name, name)


def plot_benchmark_boxplots(
    raw_df: pd.DataFrame,
    metric: str = "mAP",
    output_path: Path | None = None,
) -> None:
    """Boxplots of metric distributions per model, one subplot per dataset."""
    _ensure_dirs()
    if metric not in raw_df.columns:
        logger.warning(f"Metric '{metric}' not in results; skipping boxplots.")
        return

    if output_path is None:
        output_path = FIGURES_DIR / f"benchmark_boxplot_{metric}.pdf"

    df = raw_df.copy()
    df["model_display"] = df["model"].map(_rename_model)

    n_ds = df["dataset"].nunique()
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False)
    palette = {"AST": "#4C72B0", "MambaSpec": "#55A868", "SSAMBA": "#C44E52"}

    for i, (ds_name, ds_group) in enumerate(sorted(df.groupby("dataset"))):
        ax = axes[0, i]
        sns.boxplot(
            data=ds_group,
            x="model_display",
            y=metric,
            ax=ax,
            palette=palette,
            order=["AST", "SSAMBA", "MambaSpec"],
            width=0.5,
        )
        ax.set_title(ds_name, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(metric if i == 0 else "")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"{metric} — 5 seeds per configuration", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Benchmark boxplots saved to {output_path}")


def plot_pareto(
    summary_df: pd.DataFrame,
    profiling: dict,
    metric: str = "mAP",
    cost_key: str = "total_params",
    cost_label: str = "Parameters",
    output_path: Path | None = None,
    batch_key: str | None = None,
    higher_is_better_x: bool = False,
) -> None:
    """Scatter plot showing accuracy vs. computational cost with Pareto frontier.

    Args:
        summary_df: Aggregated results (model, dataset, metric_mean, metric_std,
            metric_ci_lower, metric_ci_upper).
        profiling: Profiling results dict keyed by display model name then dataset.
            If the dict contains a ``batch_key`` sub-level (e.g. ``"batch_1"``),
            set ``batch_key`` accordingly.
        metric: Metric column prefix.
        cost_key: Key in profiling dict for the x-axis cost
            (e.g. total_params, estimated_gflops, throughput_samples_per_sec).
        cost_label: Human-readable label for cost axis.
        output_path: Where to save.
        batch_key: Optional sub-key inside each profiling entry, e.g. ``"batch_1"``.
        higher_is_better_x: If ``True`` (e.g. throughput), higher x is better and the
            Pareto frontier runs upper-right. If ``False`` (e.g. params), lower x is
            better and the frontier runs upper-left.
    """
    _ensure_dirs()
    if output_path is None:
        output_path = FIGURES_DIR / f"pareto_{metric}_vs_{cost_key}.pdf"

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    ci_lower_col = f"{metric}_ci_lower"
    ci_upper_col = f"{metric}_ci_upper"
    has_ci = ci_lower_col in summary_df.columns and ci_upper_col in summary_df.columns

    # Map internal model names → profiling dict keys (unchanged) and paper display names
    _internal_to_profiling_key = {
        "audiospectrogramtransformer": "AST (pretrained)",
        "audiomamba": "Mamba (scratch)",
        "audiomamba_pretrained": "Mamba (SSAMBA)",
    }
    _internal_to_display = {
        "audiospectrogramtransformer": "AST",
        "audiomamba": "MambaSpec",
        "audiomamba_pretrained": "SSAMBA",
    }

    colors = {
        "AST": "#4C72B0",
        "MambaSpec": "#55A868",
        "SSAMBA": "#C44E52",
    }
    markers = {"anuraset": "o", "aswine": "s"}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect all plotted points for Pareto computation
    plot_data: list[tuple[float, float, str, str]] = []  # (cost, y, display_name, ds)

    for _, row in summary_df.iterrows():
        model_internal = row["model"]
        ds = row["dataset"]
        display_name = _internal_to_display.get(model_internal, model_internal)
        profiling_key = _internal_to_profiling_key.get(model_internal, model_internal)
        prof_entry = profiling.get(profiling_key, {}).get(ds, {})
        if batch_key is not None:
            prof_entry = prof_entry.get(batch_key, {})
        cost = prof_entry.get(cost_key, None)
        if cost is None:
            continue
        y = row[mean_col]

        if has_ci:
            yerr_lo = float(y - row[ci_lower_col])
            yerr_hi = float(row[ci_upper_col] - y)
            yerr: float | list = [[yerr_lo], [yerr_hi]]
        else:
            yerr = row[std_col]

        ax.errorbar(
            cost,
            y,
            yerr=yerr,
            fmt=markers.get(ds, "o"),
            color=colors.get(display_name, "gray"),
            markersize=10,
            capsize=4,
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{display_name}\n({ds})",
            xy=(cost, y),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=7.5,
        )
        plot_data.append((cost, y, display_name, ds))

    # --- Pareto frontier ---
    if len(plot_data) >= 2:
        coords = [(x, y) for x, y, _, _ in plot_data]

        def _is_dominated(xi: float, yi: float) -> bool:
            for xj, yj in coords:
                if higher_is_better_x:
                    if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                        return True
                else:
                    if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                        return True
            return False

        frontier_pts = sorted(
            [(x, y) for x, y in coords if not _is_dominated(x, y)],
            key=lambda p: p[0],
        )
        # Ring markers on Pareto-optimal points
        for px, py in frontier_pts:
            ax.scatter(
                px, py, s=220, facecolors="none", edgecolors="black", linewidths=1.5, zorder=4
            )
        # Connecting line along the frontier
        if len(frontier_pts) >= 2:
            fx = [p[0] for p in frontier_pts]
            fy = [p[1] for p in frontier_pts]
            ax.plot(
                fx,
                fy,
                "--",
                color="gray",
                linewidth=1.2,
                alpha=0.7,
                label="Pareto frontier",
                zorder=2,
            )
            ax.legend(fontsize=9)

    ax.set_xlabel(cost_label, fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} vs. {cost_label}", fontsize=13)
    if cost_key in ("total_params", "estimated_macs"):
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Pareto plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers for multi-function Pareto plots
# ---------------------------------------------------------------------------

_INTERNAL_TO_PROFILING_KEY: dict[str, str] = {
    "audiospectrogramtransformer": "AST (pretrained)",
    "audiomamba": "Mamba (scratch)",
    "audiomamba_pretrained": "Mamba (SSAMBA)",
}
_INTERNAL_TO_DISPLAY: dict[str, str] = {
    "audiospectrogramtransformer": "AST",
    "audiomamba": "MambaSpec",
    "audiomamba_pretrained": "SSAMBA",
}
_PARETO_COLORS: dict[str, str] = {
    "AST": "#4C72B0",
    "MambaSpec": "#55A868",
    "SSAMBA": "#C44E52",
}


def _pareto_frontier(
    coords: list[tuple[float, float]], higher_is_better_x: bool
) -> list[tuple[float, float]]:
    """Return Pareto-optimal (x, y) pairs from *coords*."""

    def _dominated(xi: float, yi: float) -> bool:
        for xj, yj in coords:
            if higher_is_better_x:
                if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                    return True
            else:
                if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                    return True
        return False

    return sorted(
        [(x, y) for x, y in coords if not _dominated(x, y)],
        key=lambda p: p[0],
    )


def _draw_pareto_ax(
    ax: plt.Axes,
    plot_data: list[tuple[float, float, list | float, str]],
    higher_is_better_x: bool = True,
    cost_key: str = "",
    label_suffix: str = "",
) -> None:
    """Draw points with CI bars, annotations, and Pareto frontier on *ax*.

    Args:
        plot_data: List of ``(x, y, yerr, display_name)`` tuples.
        higher_is_better_x: Direction of x-axis dominance.
        cost_key: Used to decide log scale.
        label_suffix: Appended to each annotation (e.g. ``" (anuraset)"``).
    """
    for x, y, yerr, display_name in plot_data:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=_PARETO_COLORS.get(display_name, "gray"),
            markersize=10,
            capsize=4,
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            f"{display_name}{label_suffix}",
            xy=(x, y),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=8,
        )

    coords = [(x, y) for x, y, _, _ in plot_data]
    if len(coords) >= 2:
        frontier_pts = _pareto_frontier(coords, higher_is_better_x)
        for px, py in frontier_pts:
            ax.scatter(
                px,
                py,
                s=220,
                facecolors="none",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )
        if len(frontier_pts) >= 2:
            fx = [p[0] for p in frontier_pts]
            fy = [p[1] for p in frontier_pts]
            ax.plot(
                fx,
                fy,
                "--",
                color="gray",
                linewidth=1.2,
                alpha=0.7,
                label="Pareto frontier",
                zorder=2,
            )
            ax.legend(fontsize=9)

    if cost_key in ("total_params", "estimated_macs"):
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)


def plot_pareto_3models(
    raw_df: pd.DataFrame,
    profiling: dict,
    metric: str = "mAP",
    cost_key: str = "throughput_samples_per_sec",
    cost_label: str = "Throughput (samples/sec, AnuraSet batch=1)",
    ref_dataset: str = "anuraset",
    batch_key: str = "batch_1",
    higher_is_better_x: bool = True,
    output_path: Path | None = None,
) -> None:
    """Architecture-level Pareto plot: 3 points, one per model.

    y-axis: grand mean mAP ± 95% bootstrap CI across all seeds × datasets.
    x-axis: cost metric at *ref_dataset* / *batch_key* (fixed input shape so
            the comparison is physically meaningful).

    Args:
        raw_df: Raw ablation results DataFrame.
        profiling: Profiling dict keyed by display model name then dataset.
        metric: Metric column prefix (``"mAP"``, ``"roc_auc_weighted"``).
        cost_key: Profiling sub-key for the x-axis.
        cost_label: Human-readable x-axis label.
        ref_dataset: Dataset whose throughput to use as the reference x-value.
        batch_key: Batch-size sub-key inside each profiling entry.
        higher_is_better_x: ``True`` for throughput; ``False`` for params/FLOPs.
        output_path: Destination PDF path.
    """
    _ensure_dirs()
    if output_path is None:
        output_path = FIGURES_DIR / f"pareto_3models_{metric}_vs_{cost_key}.pdf"

    def _bootstrap_ci(
        values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
    ) -> tuple[float, float]:
        rng = np.random.default_rng(42)
        boot_means = np.array(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
        )
        return (
            float(np.percentile(boot_means, 100 * alpha / 2)),
            float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        )

    plot_data: list[tuple[float, float, list, str]] = []
    for model_internal, group in raw_df.groupby("model"):
        vals = group[metric].dropna().values
        if len(vals) == 0:
            continue
        y = float(vals.mean())
        ci_lo, ci_hi = _bootstrap_ci(vals)
        yerr: list = [[float(y - ci_lo)], [float(ci_hi - y)]]

        display_name = _INTERNAL_TO_DISPLAY.get(model_internal, model_internal)
        profiling_key = _INTERNAL_TO_PROFILING_KEY.get(model_internal, model_internal)
        prof_entry = profiling.get(profiling_key, {}).get(ref_dataset, {})
        if batch_key:
            prof_entry = prof_entry.get(batch_key, {})
        cost = prof_entry.get(cost_key, None)
        if cost is None:
            logger.warning(f"No profiling data for {display_name} / {ref_dataset} / {batch_key}")
            continue
        plot_data.append((cost, y, yerr, display_name))

    if not plot_data:
        logger.warning("plot_pareto_3models: no data to plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    _draw_pareto_ax(ax, plot_data, higher_is_better_x=higher_is_better_x, cost_key=cost_key)
    ax.set_xlabel(cost_label, fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} vs. {cost_label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"3-model Pareto plot saved to {output_path}")


def plot_pareto_per_dataset(
    summary_df: pd.DataFrame,
    profiling: dict,
    metric: str = "mAP",
    cost_key: str = "throughput_samples_per_sec",
    cost_label: str = "Throughput (samples/sec)",
    batch_key: str = "batch_1",
    higher_is_better_x: bool = True,
    output_path: Path | None = None,
) -> None:
    """Side-by-side Pareto plots — one subplot per dataset.

    Each subplot shows 3 model points for the given dataset, with the
    Pareto frontier and 95% CI bars independently computed per dataset.

    Args:
        summary_df: Aggregated results (model, dataset, metric_mean, ci_*).
        profiling: Profiling dict keyed by display model name then dataset.
        metric: Metric column prefix.
        cost_key: Profiling sub-key for the x-axis.
        cost_label: Human-readable x-axis label.
        batch_key: Batch-size sub-key inside each profiling entry.
        higher_is_better_x: ``True`` for throughput; ``False`` for params/FLOPs.
        output_path: Destination PDF path.
    """
    _ensure_dirs()
    if output_path is None:
        output_path = FIGURES_DIR / f"pareto_per_dataset_{metric}_vs_{cost_key}.pdf"

    mean_col = f"{metric}_mean"
    ci_lower_col = f"{metric}_ci_lower"
    ci_upper_col = f"{metric}_ci_upper"
    has_ci = ci_lower_col in summary_df.columns and ci_upper_col in summary_df.columns

    datasets = sorted(summary_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)

    for col_idx, ds in enumerate(datasets):
        ax = axes[0, col_idx]
        ds_rows = summary_df[summary_df["dataset"] == ds]
        plot_data: list[tuple[float, float, list | float, str]] = []

        for _, row in ds_rows.iterrows():
            model_internal = row["model"]
            display_name = _INTERNAL_TO_DISPLAY.get(model_internal, model_internal)
            profiling_key = _INTERNAL_TO_PROFILING_KEY.get(model_internal, model_internal)
            prof_entry = profiling.get(profiling_key, {}).get(ds, {})
            if batch_key:
                prof_entry = prof_entry.get(batch_key, {})
            cost = prof_entry.get(cost_key, None)
            if cost is None:
                continue
            y = row[mean_col]
            if has_ci:
                yerr: list | float = [
                    [float(y - row[ci_lower_col])],
                    [float(row[ci_upper_col] - y)],
                ]
            else:
                yerr = row.get(f"{metric}_std", 0.0)
            plot_data.append((cost, y, yerr, display_name))

        _draw_pareto_ax(ax, plot_data, higher_is_better_x=higher_is_better_x, cost_key=cost_key)
        ax.set_title(ds, fontsize=13)
        ax.set_xlabel(cost_label, fontsize=11)
        ax.set_ylabel(metric if col_idx == 0 else "", fontsize=11)

    fig.suptitle(f"{metric} vs. {cost_label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Per-dataset Pareto plot saved to {output_path}")


def generate_benchmark_latex_table(
    summary_df: pd.DataFrame,
    profiling: dict,
    metrics: list[str] | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate a combined LaTeX table with accuracy + efficiency for the 3 models.

    Produces a table with one row per model and columns for Params, GFLOPs, and per-dataset mAP.

    Args:
        summary_df: Aggregated results.
        profiling: Profiling results dict.
        metrics: Metrics to include. Defaults to ["mAP", "roc_auc_weighted"].
        output_path: Where to save.

    Returns:
        LaTeX table string.
    """
    _ensure_dirs()
    if metrics is None:
        metrics = ["mAP", "roc_auc_weighted"]
    if output_path is None:
        output_path = REPORTS_DIR / "table_benchmark.tex"

    _internal_to_display = {
        "audiospectrogramtransformer": "AST",
        "audiomamba": "MambaSpec",
        "audiomamba_pretrained": "SSAMBA",
    }
    _internal_to_profiling_key = {
        "audiospectrogramtransformer": "AST (pretrained)",
        "audiomamba": "Mamba (scratch)",
        "audiomamba_pretrained": "Mamba (SSAMBA)",
    }

    model_order = ["audiospectrogramtransformer", "audiomamba", "audiomamba_pretrained"]
    datasets = sorted(summary_df["dataset"].unique())

    rows = []
    for model_key in model_order:
        display = _internal_to_display.get(model_key, model_key)
        # Get profiling data (use first dataset — params are the same)
        profiling_key = _internal_to_profiling_key.get(model_key, model_key)
        prof = {}
        for ds in datasets:
            prof = profiling.get(profiling_key, {}).get(ds, {})
            if prof:
                break
        params = prof.get("total_params", 0)
        params_str = f"{params / 1e6:.1f}M" if params else "—"

        row_data = {"Model": display, "Params": params_str}

        for m in metrics:
            for ds in datasets:
                mc = f"{m}_mean"
                sc = f"{m}_std"
                subset = summary_df[
                    (summary_df["model"] == model_key) & (summary_df["dataset"] == ds)
                ]
                if subset.empty:
                    row_data[f"{m}\\_{ds}"] = "—"
                else:
                    r = subset.iloc[0]
                    row_data[f"{m}\\_{ds}"] = f"{r[mc]:.4f} $\\pm$ {r[sc]:.4f}"

        rows.append(row_data)

    df = pd.DataFrame(rows).set_index("Model")
    latex = df.to_latex(escape=False, caption="Benchmark Results", label="tab:benchmark")
    output_path.write_text(latex)
    logger.info(f"Benchmark LaTeX table saved to {output_path}")
    return latex
