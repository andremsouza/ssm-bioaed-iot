"""Tests for report generator orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioaed.evaluation.report_generator import generate_report


def _create_fake_ablation(root: Path, n_seeds: int = 5) -> None:
    """Create realistic ablation directory structure with results."""
    rng = np.random.default_rng(42)
    for model in ["audiospectrogramtransformer", "audiomamba"]:
        for dataset in ["aswine"]:
            for seed in range(n_seeds):
                d = root / model / dataset / f"seed_{seed}"
                d.mkdir(parents=True)
                result = {
                    "test_metrics": {
                        "mAP": float(rng.normal(0.6, 0.02)),
                        "roc_auc_weighted": float(rng.normal(0.7, 0.03)),
                    }
                }
                (d / "results.json").write_text(json.dumps(result))


class TestGenerateReport:
    def test_generates_outputs(self, tmp_path: Path) -> None:
        ablation_root = tmp_path / "ablation"
        ablation_root.mkdir()
        _create_fake_ablation(ablation_root, n_seeds=5)

        reports_dir = tmp_path / "reports"
        with (
            patch("bioaed.evaluation.report_generator.REPORTS_DIR", reports_dir),
            patch("bioaed.evaluation.visualizations.REPORTS_DIR", reports_dir),
            patch("bioaed.evaluation.visualizations.FIGURES_DIR", reports_dir / "figures"),
        ):
            generate_report(ablation_root=ablation_root, metrics=["mAP"])

        assert (reports_dir / "stats_mAP.txt").exists()

    def test_empty_ablation_skips(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        with patch("bioaed.evaluation.report_generator.REPORTS_DIR", reports_dir):
            generate_report(ablation_root=tmp_path / "empty", metrics=["mAP"])
        # No reports should be generated
        assert not reports_dir.exists() or not list(reports_dir.iterdir())

    def test_missing_metric_skipped(self, tmp_path: Path) -> None:
        ablation_root = tmp_path / "ablation"
        ablation_root.mkdir()
        _create_fake_ablation(ablation_root, n_seeds=5)

        reports_dir = tmp_path / "reports"
        with (
            patch("bioaed.evaluation.report_generator.REPORTS_DIR", reports_dir),
            patch("bioaed.evaluation.visualizations.REPORTS_DIR", reports_dir),
            patch("bioaed.evaluation.visualizations.FIGURES_DIR", reports_dir / "figures"),
        ):
            generate_report(ablation_root=ablation_root, metrics=["nonexistent"])
        # No report for this metric
        assert not (reports_dir / "stats_nonexistent.txt").exists()
