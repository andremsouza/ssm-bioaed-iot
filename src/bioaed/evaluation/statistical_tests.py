"""Non-parametric statistical comparison of ablation results.

Implements the standard workflow for comparing classifiers over multiple
datasets/seeds as recommended by Demsar (2006) and Herbold (2020):

    1. Shapiro-Wilk normality test per population.
    2. Friedman omnibus test (>2 populations).
    3. Nemenyi or Wilcoxon-Holm post-hoc.
    4. Effect sizes (Cohen's d for paired samples).
    5. Automated selection using ``autorank``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import autorank
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class StatisticalResult:
    """Container for statistical test results."""

    metric: str
    n_configs: int
    n_observations: int
    normality: dict[str, float] = field(default_factory=dict)
    omnibus_test: str = ""
    omnibus_stat: float = 0.0
    omnibus_pvalue: float = 1.0
    posthoc_method: str = ""
    posthoc_table: pd.DataFrame | None = None
    effect_sizes: pd.DataFrame | None = None
    autorank_result: autorank.RankResult | None = None

    @property
    def significant(self) -> bool:
        return self.omnibus_pvalue < 0.05


def _shapiro_wilk_per_column(df: pd.DataFrame) -> dict[str, float]:
    """Shapiro-Wilk normality test for each configuration."""
    results = {}
    for col in df.columns:
        vals = df[col].dropna().values
        if len(vals) >= 3:
            _, pval = stats.shapiro(vals)
            results[col] = float(pval)
        else:
            results[col] = float("nan")
    return results


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (using pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Cohen's d effect sizes between all configs."""
    cols = df.columns.tolist()
    n = len(cols)
    effect_matrix = pd.DataFrame(0.0, index=cols, columns=cols)
    for i in range(n):
        for j in range(i + 1, n):
            a = df[cols[i]].dropna().values
            b = df[cols[j]].dropna().values
            d = _cohens_d(a, b)
            effect_matrix.loc[cols[i], cols[j]] = d
            effect_matrix.loc[cols[j], cols[i]] = -d
    return effect_matrix


def run_statistical_comparison(
    wide_df: pd.DataFrame,
    metric: str = "mAP",
    alpha: float = 0.05,
) -> StatisticalResult:
    """Run full statistical comparison pipeline.

    Args:
        wide_df: Wide-format DataFrame where each column is a configuration
            and each row is one measurement (seed/fold). Produced by
            :func:`bioaed.analysis.build_comparison_matrix`.
        metric: Name of the metric being compared (for reporting).
        alpha: Significance level.

    Returns:
        :class:`StatisticalResult` with all test outcomes.
    """
    df = wide_df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    n_configs = len(df.columns)
    n_obs = len(df)

    result = StatisticalResult(
        metric=metric,
        n_configs=n_configs,
        n_observations=n_obs,
    )

    if n_configs < 2:
        logger.warning("Need ≥2 configs to compare; skipping statistical tests.")
        return result

    # 1. Normality
    result.normality = _shapiro_wilk_per_column(df)
    all_normal = all(p > alpha for p in result.normality.values() if not np.isnan(p))
    normality_msg = "all normal" if all_normal else "non-normal distributions detected"
    logger.info(f"Normality (Shapiro-Wilk): {normality_msg}")

    # 2. Omnibus test
    if n_configs == 2:
        col_a, col_b = df.columns[0], df.columns[1]
        stat, pval = stats.wilcoxon(df[col_a], df[col_b])
        result.omnibus_test = "Wilcoxon signed-rank"
        result.omnibus_stat = float(stat)
        result.omnibus_pvalue = float(pval)
    else:
        stat, pval = stats.friedmanchisquare(*[df[col].values for col in df.columns])
        result.omnibus_test = "Friedman"
        result.omnibus_stat = float(stat)
        result.omnibus_pvalue = float(pval)

    logger.info(
        f"{result.omnibus_test}: stat={result.omnibus_stat:.4f}, p={result.omnibus_pvalue:.6f}"
    )

    # 3. Post-hoc (only if omnibus significant and >2 configs)
    if result.significant and n_configs > 2:
        import scikit_posthocs as sp

        posthoc_df = sp.posthoc_nemenyi_friedman(df)
        result.posthoc_method = "Nemenyi"
        result.posthoc_table = posthoc_df
        logger.info("Post-hoc: Nemenyi test applied")

    # 4. Effect sizes
    result.effect_sizes = compute_effect_sizes(df)

    # 5. autorank automated ranking
    try:
        ar_result = autorank.autorank(df, alpha=alpha, verbose=False)
        result.autorank_result = ar_result
        logger.info("autorank analysis complete")
    except Exception as e:
        logger.warning(f"autorank failed: {e}")

    return result


def format_results_text(result: StatisticalResult) -> str:
    """Format statistical results as human-readable text."""
    lines = [
        f"=== Statistical Comparison: {result.metric} ===",
        f"Configurations: {result.n_configs} | Observations: {result.n_observations}",
        f"Omnibus test: {result.omnibus_test} "
        f"(stat={result.omnibus_stat:.4f}, p={result.omnibus_pvalue:.6f})",
        f"Significant: {result.significant}",
    ]

    if result.posthoc_table is not None:
        lines.append(f"\nPost-hoc ({result.posthoc_method}):")
        lines.append(result.posthoc_table.to_string())

    if result.effect_sizes is not None:
        lines.append("\nEffect sizes (Cohen's d):")
        lines.append(result.effect_sizes.to_string(float_format="%.3f"))

    return "\n".join(lines)
