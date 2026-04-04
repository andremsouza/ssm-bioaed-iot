"""Tests for Hydra entry-point modules (import coverage + helper logic)."""

from __future__ import annotations


class TestEvaluateImport:
    def test_module_imports(self) -> None:
        import bioaed.evaluate  # noqa: F401


class TestTrainImport:
    def test_module_imports(self) -> None:
        import bioaed.train  # noqa: F401


class TestSweepImport:
    def test_module_imports(self) -> None:
        import bioaed.sweep  # noqa: F401
