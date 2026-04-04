"""Tests for statistical comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioaed.evaluation.statistical_tests import (
    StatisticalResult,
    _cohens_d,
    _shapiro_wilk_per_column,
    compute_effect_sizes,
    format_results_text,
    run_statistical_comparison,
)


def _make_wide_df(n_configs: int = 4, n_obs: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_configs):
        data[f"config_{i}"] = rng.normal(loc=0.5 + i * 0.05, scale=0.02, size=n_obs)
    return pd.DataFrame(data)


class TestShapiroWilk:
    def test_normal_data(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
        result = _shapiro_wilk_per_column(df)
        assert "a" in result and "b" in result
        # Normal data should typically have p > 0.05
        assert result["a"] > 0.01

    def test_too_few_samples(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = _shapiro_wilk_per_column(df)
        assert np.isnan(result["a"])


class TestCohensD:
    def test_identical_distributions(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = _cohens_d(a, a)
        assert abs(d) < 1e-6

    def test_large_effect(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        d = _cohens_d(a, b)
        assert abs(d) > 1.0  # large effect

    def test_too_few_samples(self) -> None:
        d = _cohens_d(np.array([1.0]), np.array([2.0]))
        assert d == 0.0


class TestComputeEffectSizes:
    def test_symmetric(self) -> None:
        df = _make_wide_df(n_configs=3, n_obs=15)
        es = compute_effect_sizes(df)
        # Diagonal should be 0
        for col in df.columns:
            assert abs(es.loc[col, col]) < 1e-6
        # Antisymmetric
        assert abs(es.loc["config_0", "config_1"] + es.loc["config_1", "config_0"]) < 1e-6


class TestRunStatisticalComparison:
    def test_four_configs(self) -> None:
        df = _make_wide_df(n_configs=4, n_obs=10)
        result = run_statistical_comparison(df, metric="test_metric")
        assert isinstance(result, StatisticalResult)
        assert result.metric == "test_metric"
        assert result.n_configs == 4
        assert result.n_observations == 10
        assert result.omnibus_test == "Friedman"
        assert result.effect_sizes is not None
        assert result.autorank_result is not None

    def test_two_configs_uses_wilcoxon(self) -> None:
        df = _make_wide_df(n_configs=2, n_obs=10)
        result = run_statistical_comparison(df, metric="mAP")
        assert result.omnibus_test == "Wilcoxon signed-rank"

    def test_single_config_skips(self) -> None:
        df = pd.DataFrame({"a": [0.5, 0.6, 0.7]})
        result = run_statistical_comparison(df, metric="mAP")
        assert result.n_configs == 1
        assert result.omnibus_pvalue == 1.0

    def test_significant_result_has_posthoc(self) -> None:
        """When configs are clearly different, posthoc should be generated."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "bad": rng.normal(0.3, 0.01, 15),
                "mid": rng.normal(0.5, 0.01, 15),
                "good": rng.normal(0.8, 0.01, 15),
            }
        )
        result = run_statistical_comparison(df, metric="mAP")
        assert result.significant
        assert result.posthoc_table is not None
        assert result.posthoc_method == "Nemenyi"


class TestFormatResultsText:
    def test_output_is_string(self) -> None:
        df = _make_wide_df(n_configs=3, n_obs=10)
        result = run_statistical_comparison(df, metric="mAP")
        text = format_results_text(result)
        assert "mAP" in text
        assert "Friedman" in text
