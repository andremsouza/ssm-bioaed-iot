"""Structured logging with Loguru and Rich."""

from __future__ import annotations

import logging
import sys
from typing import Any

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def configure_logging(level: str = "INFO") -> None:
    """Configure Loguru to use Rich for console output and intercept stdlib logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Remove default Loguru handler
    logger.remove()

    # Add Rich-formatted console handler
    logger.add(
        sink=sys.stderr,
        level=level,
        format="{message}",
        diagnose=True,
        backtrace=True,
    )

    # Intercept stdlib logging → route through Loguru
    logging.basicConfig(
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        level=getattr(logging, level),
        format="%(message)s",
        force=True,
    )

    logger.info(f"Logging configured at level={level}")


def log_config(cfg: dict[str, Any]) -> None:
    """Pretty-print a configuration dictionary using Rich.

    Args:
        cfg: Configuration dictionary to display.
    """
    from rich.panel import Panel
    from rich.pretty import Pretty

    console.print(Panel(Pretty(cfg), title="[bold]Experiment Config[/bold]", expand=False))
