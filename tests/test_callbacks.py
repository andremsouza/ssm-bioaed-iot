"""Tests for training callbacks."""

from __future__ import annotations

from bioaed.training.callbacks import EarlyStopping, MetricLogger, RichProgressCallback


class TestEarlyStopping:
    def test_no_stop_on_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        assert es.step(1.0) is False
        assert es.step(0.9) is False
        assert es.step(0.8) is False

    def test_stops_after_patience(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)  # best
        es.step(1.1)  # no improvement
        es.step(1.1)  # no improvement
        assert es.step(1.1) is True  # 3rd miss → stop

    def test_resets_counter_on_improvement(self) -> None:
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.1)  # miss 1
        es.step(1.1)  # miss 2
        es.step(0.5)  # improvement → reset
        assert es.counter == 0
        assert es.best_value == 0.5

    def test_min_delta(self) -> None:
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)  # best = 1.0
        es.step(0.95)  # 1.0 - 0.95 = 0.05 < 0.1 → no improvement
        assert es.step(0.96) is True  # 2nd miss → stop

    def test_exact_patience_boundary(self) -> None:
        es = EarlyStopping(patience=1)
        es.step(1.0)
        assert es.step(1.0) is True  # first miss → stop immediately


class TestRichProgressCallback:
    def test_log_epoch_runs(self) -> None:
        cb = RichProgressCallback()
        # Should not raise
        cb.log_epoch({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "lr": 1e-3})

    def test_log_epoch_handles_non_float(self) -> None:
        cb = RichProgressCallback()
        cb.log_epoch({"epoch": 1, "status": "ok"})


class TestMetricLogger:
    def test_stores_history(self) -> None:
        ml = MetricLogger()
        ml.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "lr": 1e-3})
        ml.log({"epoch": 2, "train_loss": 0.3, "val_loss": 0.25, "lr": 5e-4})
        assert len(ml.history) == 2
        assert ml.history[0]["epoch"] == 1
        assert ml.history[1]["val_loss"] == 0.25

    def test_empty_history_initially(self) -> None:
        ml = MetricLogger()
        assert ml.history == []
