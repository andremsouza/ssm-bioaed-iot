"""Tests for visualization and report generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bioaed.evaluation.statistical_tests import StatisticalResult, run_statistical_comparison
from bioaed.evaluation.visualizations import (
    generate_latex_table,
    plot_cd_diagram,
    plot_metric_boxplots,
)


@pytest.fixture
def summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["inceptiontime", "ast"],
            "dataset": ["aswine", "aswine"],
            "mAP_mean": [0.65, 0.70],
            "mAP_std": [0.02, 0.01],
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model in ["inceptiontime", "ast"]:
        for seed in range(5):
            rows.append(
                {
                    "model": model,
                    "dataset": "aswine",
                    "seed": seed,
                    "mAP": float(rng.normal(0.6, 0.02)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def stat_result() -> StatisticalResult:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "config_a": rng.normal(0.3, 0.01, 15),
            "config_b": rng.normal(0.5, 0.01, 15),
            "config_c": rng.normal(0.7, 0.01, 15),
        }
    )
    return run_statistical_comparison(df, metric="mAP")


class TestPlotCdDiagram:
    def test_saves_file(self, stat_result: StatisticalResult, tmp_path: Path) -> None:
        out = tmp_path / "cd.pdf"
        plot_cd_diagram(stat_result, output_path=out)
        assert out.exists()

    def test_no_autorank_skips(self, tmp_path: Path) -> None:
        result = StatisticalResult(metric="mAP", n_configs=0, n_observations=0)
        out = tmp_path / "cd.pdf"
        plot_cd_diagram(result, output_path=out)
        assert not out.exists()


class TestPlotMetricBoxplots:
    def test_saves_file(self, raw_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "boxplot.pdf"
        plot_metric_boxplots(raw_df, metric="mAP", output_path=out)
        assert out.exists()

    def test_missing_metric_skips(self, raw_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "boxplot.pdf"
        plot_metric_boxplots(raw_df, metric="nonexistent", output_path=out)
        assert not out.exists()


class TestGenerateLatexTable:
    def test_returns_latex_string(self, summary_df: pd.DataFrame, tmp_path: Path) -> None:
        out = tmp_path / "table.tex"
        latex = generate_latex_table(summary_df, metric="mAP", output_path=out)
        assert "\\begin{tabular}" in latex
        assert out.exists()
