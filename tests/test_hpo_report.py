"""Tests for HPO partial report generator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bioaed.hpo.report import (
    StudySummary,
    _build_markdown,
    _load_history,
    _load_study,
    generate_hpo_report,
)


@pytest.fixture()
def fake_hpo_dir(tmp_path: Path) -> Path:
    """Create a minimal fake HPO directory with two studies."""
    for model, dataset, n_trials in [
        ("inceptiontime", "aswine", 3),
        ("audiomamba", "anuraset", 2),
    ]:
        study_dir = tmp_path / model / dataset
        study_dir.mkdir(parents=True)
        db = study_dir / "study.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, state TEXT)")
            conn.execute("CREATE TABLE trial_values (trial_id INTEGER, value REAL)")
            for i in range(n_trials):
                conn.execute("INSERT INTO trials VALUES (?, 'COMPLETE')", (i,))
                conn.execute("INSERT INTO trial_values VALUES (?, ?)", (i, 0.5 - i * 0.1))
        params = {"learning_rate": 1e-3, "batch_size": 32}
        (study_dir / "best_params.json").write_text(json.dumps(params))
    return tmp_path


class TestLoadStudy:
    def test_reads_completed_count(self, fake_hpo_dir: Path) -> None:
        db = fake_hpo_dir / "inceptiontime" / "aswine" / "study.db"
        s = _load_study(db)
        assert s.completed == 3
        assert s.total == 3
        assert s.model_key == "inceptiontime"
        assert s.dataset == "aswine"

    def test_reads_best_val_loss(self, fake_hpo_dir: Path) -> None:
        db = fake_hpo_dir / "inceptiontime" / "aswine" / "study.db"
        s = _load_study(db)
        assert s.best_val_loss == pytest.approx(0.3)

    def test_reads_best_params(self, fake_hpo_dir: Path) -> None:
        db = fake_hpo_dir / "inceptiontime" / "aswine" / "study.db"
        s = _load_study(db)
        assert s.best_params["batch_size"] == 32

    def test_missing_best_params_json(self, fake_hpo_dir: Path) -> None:
        db = fake_hpo_dir / "audiomamba" / "anuraset" / "study.db"
        (fake_hpo_dir / "audiomamba" / "anuraset" / "best_params.json").unlink()
        s = _load_study(db)
        assert s.best_params == {}


class TestLoadHistory:
    def test_returns_dataframe_with_best_so_far(self, fake_hpo_dir: Path) -> None:
        db = fake_hpo_dir / "inceptiontime" / "aswine" / "study.db"
        df = _load_history(db)
        assert list(df.columns) == ["trial_id", "value", "best_so_far"]
        assert len(df) == 3
        # best_so_far is monotonically decreasing
        assert df["best_so_far"].is_monotonic_decreasing


class TestBuildMarkdown:
    def test_contains_model_and_dataset(self) -> None:
        s = StudySummary(
            model_key="inceptiontime",
            dataset="aswine",
            completed=10,
            total=30,
            best_val_loss=0.05,
            best_trial_id=7,
            best_params={"learning_rate": 1e-3},
        )
        md = _build_markdown([s])
        assert "InceptionTime" in md
        assert "aswine" in md
        assert "0.050000" in md
        assert "learning_rate" in md


class TestGenerateHpoReport:
    def test_creates_report_file(self, fake_hpo_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "hpo_report.md"
        summaries = generate_hpo_report(hpo_root=fake_hpo_dir, output_path=out, plot=False)
        assert out.exists()
        assert len(summaries) == 2

    def test_returns_empty_on_no_studies(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        summaries = generate_hpo_report(
            hpo_root=empty_dir, output_path=tmp_path / "r.md", plot=False
        )
        assert summaries == []
