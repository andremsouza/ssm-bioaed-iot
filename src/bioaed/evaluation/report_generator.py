"""Orchestrate result analysis: load → test → visualize → report.

Entry point for the full analysis pipeline after ablation runs complete.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from bioaed.analysis import (
    ABLATION_ROOT,
    aggregate_results,
    build_comparison_matrix,
    load_ablation_results,
)
from bioaed.evaluation.statistical_tests import (
    format_results_text,
    run_statistical_comparison,
)
from bioaed.evaluation.visualizations import (
    generate_latex_table,
    plot_ablation_heatmap,
    plot_cd_diagram,
    plot_metric_boxplots,
)

REPORTS_DIR = Path("reports")


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
        plot_ablation_heatmap(summary, metric=metric)
        plot_metric_boxplots(raw_df, metric=metric)
        generate_latex_table(summary, metric=metric)

    logger.success("Report generation complete. Check reports/ directory.")


if __name__ == "__main__":
    generate_report()
