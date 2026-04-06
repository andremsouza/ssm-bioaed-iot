"""Tests for result aggregation and analysis."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bioaed.analysis import (
    aggregate_results,
    build_comparison_matrix,
    compute_dcqg_delta,
    load_ablation_results,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Simulate raw ablation results."""
    rows = []
    rng = np.random.default_rng(42)
    for model in ["inceptiontime", "ast"]:
        for dataset in ["aswine", "anuraset"]:
            for dcqg in ["dcqg_on", "dcqg_off"]:
                for seed in range(5):
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "dcqg": dcqg,
                            "seed": seed,
                            "fold": None,
                            "mAP": rng.normal(0.6 if dcqg == "dcqg_on" else 0.55, 0.02),
                            "roc_auc_weighted": rng.normal(0.7, 0.03),
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def ablation_dir(tmp_path: Path) -> Path:
    """Create a fake ablation output directory with JSON results."""
    rng = np.random.default_rng(42)
    for model in ["inceptiontime"]:
        for dataset in ["aswine"]:
            for dcqg in ["dcqg_on", "dcqg_off"]:
                for seed in range(3):
                    out_dir = tmp_path / model / dataset / dcqg / f"seed_{seed}"
                    out_dir.mkdir(parents=True)
                    result = {
                        "test_metrics": {
                            "mAP": float(rng.normal(0.6, 0.02)),
                            "roc_auc_weighted": float(rng.normal(0.7, 0.03)),
                        }
                    }
                    (out_dir / "results.json").write_text(json.dumps(result))
    return tmp_path


class TestLoadAblationResults:
    def test_loads_json_files(self, ablation_dir: Path) -> None:
        df = load_ablation_results(ablation_dir)
        assert len(df) == 6  # 1 model × 1 dataset × 2 dcqg × 3 seeds
        assert "mAP" in df.columns
        assert "model" in df.columns

    def test_empty_dir(self, tmp_path: Path) -> None:
        df = load_ablation_results(tmp_path)
        assert df.empty


class TestAggregateResults:
    def test_aggregate_produces_summary(self, sample_raw_df: pd.DataFrame) -> None:
        summary = aggregate_results(sample_raw_df, metric="mAP")
        assert "mAP_mean" in summary.columns
        assert "mAP_std" in summary.columns
        assert "mAP_ci_lower" in summary.columns
        assert "mAP_ci_upper" in summary.columns
        assert "n_runs" in summary.columns
        # 2 models × 2 datasets × 2 dcqg = 8 configs
        assert len(summary) == 8

    def test_ci_bounds(self, sample_raw_df: pd.DataFrame) -> None:
        summary = aggregate_results(sample_raw_df, metric="mAP")
        for _, row in summary.iterrows():
            assert row["mAP_ci_lower"] <= row["mAP_mean"]
            assert row["mAP_ci_upper"] >= row["mAP_mean"]


class TestComputeDcqgDelta:
    def test_delta_columns(self, sample_raw_df: pd.DataFrame) -> None:
        summary = aggregate_results(sample_raw_df, metric="mAP")
        delta = compute_dcqg_delta(summary, metric="mAP")
        assert "model" in delta.columns
        assert "dataset" in delta.columns
        assert "delta" in delta.columns


class TestBuildComparisonMatrix:
    def test_wide_format(self, sample_raw_df: pd.DataFrame) -> None:
        wide = build_comparison_matrix(sample_raw_df, metric="mAP")
        # Columns should be config strings
        assert wide.shape[1] > 0
        # Each column is a config
        assert all("_" in col for col in wide.columns)

    def test_with_folds(self) -> None:
        """Data with fold column should include fold in index."""
        rng = np.random.default_rng(42)
        rows = []
        for seed in range(3):
            for fold in range(2):
                for dcqg in ["dcqg_on", "dcqg_off"]:
                    rows.append(
                        {
                            "model": "inceptiontime",
                            "dataset": "aswine",
                            "dcqg": dcqg,
                            "seed": seed,
                            "fold": fold,
                            "mAP": float(rng.normal(0.6, 0.02)),
                        }
                    )
        df = pd.DataFrame(rows)
        wide = build_comparison_matrix(df, metric="mAP")
        assert wide.shape[0] > 0


class TestLoadFoldResults:
    def test_fold_structure(self, tmp_path: Path) -> None:
        """results.json inside fold_N subdir should be loaded with fold index."""
        rng = np.random.default_rng(0)
        model_dir = tmp_path / "inceptiontime" / "aswine" / "dcqg_on" / "seed_0"
        for fold in range(2):
            fold_dir = model_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            result = {"test_metrics": {"mAP": float(rng.normal(0.6, 0.01))}}
            (fold_dir / "results.json").write_text(json.dumps(result))

        df = load_ablation_results(tmp_path)
        assert len(df) == 2
        assert df["fold"].notna().all()
        assert set(df["fold"]) == {0, 1}


class TestAggregateEmpty:
    def test_empty_metric(self) -> None:
        """Aggregation with missing metric should skip gracefully."""
        df = pd.DataFrame(
            {
                "model": ["a", "a"],
                "dataset": ["ds", "ds"],
                "dcqg": ["on", "on"],
                "mAP": [np.nan, np.nan],
            }
        )
        summary = aggregate_results(df, metric="mAP")
        assert len(summary) == 0


class TestAnalysisMainBlock:
    """Covers the ``if __name__ == '__main__'`` block in analysis.py."""

    def test_main_block_runs_with_existing_results(self) -> None:
        """Running analysis as __main__ should call load/aggregate/print without error."""
        # Suppress stdout from print() calls inside the __main__ block
        with patch("builtins.print"):
            # runpy executes the if __name__ == "__main__" block
            runpy.run_module("bioaed.analysis", run_name="__main__", alter_sys=False)

