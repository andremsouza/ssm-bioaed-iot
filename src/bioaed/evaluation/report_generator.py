"""Orchestrate result analysis: load → test → visualize → report.

Entry point for the full analysis pipeline after ablation runs complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from bioaed.analysis import (
    ABLATION_ROOT,
    aggregate_results,
    build_comparison_matrix,
    build_model_comparison_matrix,
    load_ablation_results,
)
from bioaed.evaluation.statistical_tests import (
    format_results_text,
    run_statistical_comparison,
)
from bioaed.evaluation.visualizations import (
    generate_benchmark_latex_table,
    generate_latex_table,
    plot_benchmark_boxplots,
    plot_cd_diagram,
    plot_metric_boxplots,
    plot_pareto,
    plot_pareto_3models,
    plot_pareto_per_dataset,
)

REPORTS_DIR = Path("reports")

# Models to include in the benchmark paper
_BENCHMARK_MODELS = {
    "audiospectrogramtransformer",
    "audiomamba",
    "audiomamba_pretrained",
}


def generate_report(
    ablation_root: Path = ABLATION_ROOT,
    metrics: list[str] | None = None,
) -> None:
    """Run the full analysis and report generation pipeline.

    Args:
        ablation_root: Root directory of ablation outputs.
        metrics: List of metrics to analyse. Defaults to ``["mAP", "roc_auc_weighted"]``.
    """
    if metrics is None:
        metrics = ["mAP", "roc_auc_weighted"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load results
    raw_df = load_ablation_results(ablation_root)

    if raw_df.empty:
        logger.error("No ablation results found. Run ablation first.")
        return

    # Filter to benchmark models only
    raw_df = raw_df[raw_df["model"].isin(_BENCHMARK_MODELS)].copy()
    logger.info(
        f"Benchmark models: {sorted(raw_df['model'].unique().tolist())}, {len(raw_df)} rows"
    )

    for metric in metrics:
        if metric not in raw_df.columns:
            logger.warning(f"Metric '{metric}' not in results; skipping.")
            continue

        logger.info(f"--- Analysing metric: {metric} ---")

        # 2. Aggregate
        summary = aggregate_results(raw_df, metric=metric)

        # 3. Comparison matrix for statistical tests
        wide = build_comparison_matrix(raw_df, metric=metric)

        # 4. Statistical tests
        stat_result = run_statistical_comparison(wide, metric=metric)
        report_text = format_results_text(stat_result)
        report_path = REPORTS_DIR / f"stats_{metric}.txt"
        report_path.write_text(report_text)
        logger.info(f"Statistical report saved to {report_path}")

        # 5. Visualizations
        plot_cd_diagram(stat_result)
        plot_metric_boxplots(raw_df, metric=metric)
        generate_latex_table(summary, metric=metric)

    logger.success("Report generation complete. Check reports/ directory.")


def _load_benchmark_df(
    ablation_root: Path = ABLATION_ROOT,
) -> pd.DataFrame:
    """Load results for the 3 benchmark models."""
    raw_df = load_ablation_results(ablation_root)

    # Filter: only benchmark models
    raw_df = raw_df[raw_df["model"].isin(_BENCHMARK_MODELS)].copy()

    logger.info(f"Benchmark data: {len(raw_df)} rows, models={raw_df['model'].unique().tolist()}")
    return raw_df


def _build_per_dataset_wide(
    df: pd.DataFrame,
    metric: str,
    dataset: str,
) -> pd.DataFrame:
    """Build wide-format matrix for a single dataset (columns = models, rows = seeds)."""
    subset = df[df["dataset"] == dataset].copy()
    return subset.pivot_table(index="seed", columns="model", values=metric)


def generate_benchmark_report(
    ablation_root: Path = ABLATION_ROOT,
    metrics: list[str] | None = None,
    profiling_path: Path | None = None,
) -> None:
    """Run the 3-config benchmark analysis pipeline.

    Produces statistical reports, CD diagrams, boxplots, Pareto plots,
    and a combined LaTeX table for the SBBD 2026 short paper.

    Args:
        ablation_root: Root directory of ablation outputs.
        metrics: Metrics to analyse.
        profiling_path: Path to profiling_results.json.
    """
    if metrics is None:
        metrics = ["mAP", "roc_auc_weighted"]
    if profiling_path is None:
        profiling_path = REPORTS_DIR / "profiling_results_extended.json"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = _load_benchmark_df(ablation_root)

    if raw_df.empty:
        logger.error("No benchmark results found.")
        return

    # Load profiling data
    profiling: dict = {}
    if profiling_path.exists():
        with open(profiling_path) as f:
            profiling = json.load(f)
        logger.info(f"Loaded profiling data from {profiling_path}")

    datasets = sorted(raw_df["dataset"].unique())

    for metric in metrics:
        if metric not in raw_df.columns:
            logger.warning(f"Metric '{metric}' not in results; skipping.")
            continue

        logger.info(f"--- Benchmark analysis: {metric} ---")

        # 1. Aggregate per model × dataset
        summary = aggregate_results(raw_df, metric=metric)

        # 2. Statistical tests — per dataset (3 models, 5 seeds each)
        for ds in datasets:
            wide = _build_per_dataset_wide(raw_df, metric, ds)
            if wide.shape[1] < 2:
                continue
            stat_result = run_statistical_comparison(wide, metric=f"{metric}_{ds}")
            report_text = format_results_text(stat_result)
            report_path = REPORTS_DIR / f"benchmark_stats_{metric}_{ds}.txt"
            report_path.write_text(report_text)
            logger.info(f"Stats saved to {report_path}")

            # CD diagram per dataset
            plot_cd_diagram(stat_result)

        # 3a. Combined comparison across datasets (auxiliary — 6-treatment matrix)
        wide_all = build_comparison_matrix(raw_df, metric=metric)
        stat_all = run_statistical_comparison(wide_all, metric=f"{metric}_all")
        report_path = REPORTS_DIR / f"benchmark_stats_{metric}_all.txt"
        report_path.write_text(format_results_text(stat_all))
        plot_cd_diagram(stat_all)

        # 3b. Paper CD diagram: 3 models × 10 blocks (Demšar 2006 formulation)
        # Rows = (dataset, seed) pairs → 10 blocks; columns = 3 models.
        # This tests whether model differences hold consistently across all
        # (dataset, seed) combinations, which is the primary research question.
        wide_model = build_model_comparison_matrix(raw_df, metric=metric)
        stat_model = run_statistical_comparison(wide_model, metric=metric)
        stats_path = REPORTS_DIR / f"stats_{metric}.txt"
        stats_path.write_text(format_results_text(stat_model))
        logger.info(f"Paper stats saved to {stats_path}")
        plot_cd_diagram(stat_model)  # saves to reports/figures/cd_{metric}.pdf

        # 4. Boxplots (benchmark variant)
        plot_benchmark_boxplots(raw_df, metric=metric)

        # 5. Pareto plots
        if profiling:
            # 5a. Original 6-point plots (model × dataset, supplementary)
            plot_pareto(
                summary,
                profiling,
                metric=metric,
                cost_key="total_params",
                cost_label="Parameters (log scale)",
                batch_key="batch_1",
                higher_is_better_x=False,
            )
            plot_pareto(
                summary,
                profiling,
                metric=metric,
                cost_key="estimated_gflops",
                cost_label="GFLOPs",
                batch_key="batch_1",
                higher_is_better_x=False,
            )
            plot_pareto(
                summary,
                profiling,
                metric=metric,
                cost_key="throughput_samples_per_sec",
                cost_label="Throughput (samples/sec)",
                batch_key="batch_1",
                higher_is_better_x=True,
            )
            # 5b. Paper figure: 3 architecture points, grand mean across datasets
            #     x = AnuraSet batch=1 throughput (fixed reference input shape)
            plot_pareto_3models(
                raw_df,
                profiling,
                metric=metric,
                cost_key="throughput_samples_per_sec",
                cost_label="Throughput (samples/sec, AnuraSet batch=1)",
                ref_dataset="anuraset",
                batch_key="batch_1",
                higher_is_better_x=True,
            )
            # 5c. Side-by-side per-dataset subplots — throughput, params, GFLOPs
            plot_pareto_per_dataset(
                summary,
                profiling,
                metric=metric,
                cost_key="throughput_samples_per_sec",
                cost_label="Throughput (samples/sec)",
                batch_key="batch_1",
                higher_is_better_x=True,
            )
            plot_pareto_per_dataset(
                summary,
                profiling,
                metric=metric,
                cost_key="total_params",
                cost_label="Parameters",
                batch_key="batch_1",
                higher_is_better_x=False,
            )
            plot_pareto_per_dataset(
                summary,
                profiling,
                metric=metric,
                cost_key="estimated_gflops",
                cost_label="GFLOPs",
                batch_key="batch_1",
                higher_is_better_x=False,
            )

    # 6. Combined LaTeX table with profiling.
    # Keep only unique model/dataset rows with all metric columns.
    summary_merged = aggregate_results(raw_df, metric=metrics[0])
    for m in metrics[1:]:
        if m not in raw_df.columns:
            continue
        extra = aggregate_results(raw_df, metric=m)
        extra = extra.drop(columns=["n_runs"], errors="ignore")
        summary_merged = summary_merged.merge(extra, on=["model", "dataset"], how="outer")

    generate_benchmark_latex_table(summary_merged, profiling, metrics=metrics)

    logger.success("Benchmark report generation complete. Check reports/ directory.")


if __name__ == "__main__":
    generate_report()
