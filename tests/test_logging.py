"""Tests for logging utilities."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from bioaed.utils.logging import configure_logging, log_config


class TestConfigureLogging:
    def test_default_level(self) -> None:
        # Should not raise
        configure_logging()

    def test_debug_level(self) -> None:
        configure_logging(level="DEBUG")

    def test_warning_level(self) -> None:
        configure_logging(level="WARNING")


class TestLogConfig:
    def test_prints_config(self) -> None:
        cfg = {"seed": 42, "model": "inceptiontime", "nested": {"lr": 0.001}}
        # Should not raise; just verify it runs
        log_config(cfg)

    def test_empty_config(self) -> None:
        log_config({})
