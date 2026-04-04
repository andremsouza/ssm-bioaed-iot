"""Training callbacks: early stopping, progress display, and metric logging."""

from __future__ import annotations

from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()


class EarlyStopping:
    """Patience-based early stopping monitor.

    Monitors a metric (lower is better) and signals to stop training
    when no improvement is observed for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float = float("inf")
        self.counter: int = 0

    def step(self, value: float) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value (lower is better).

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class RichProgressCallback:
    """Rich-based progress display for training epochs.

    Provides pretty-printed epoch summaries in the terminal.
    """

    def log_epoch(self, metrics: dict[str, Any]) -> None:
        """Display epoch metrics in a Rich table.

        Args:
            metrics: Dictionary of metric names → values.
        """
        table = Table(title=f"Epoch {metrics.get('epoch', '?')}", show_lines=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in metrics.items():
            if key == "epoch":
                continue
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))

        console.print(table)


class MetricLogger:
    """Loguru-based metric logger that stores history for analysis.

    Logs metrics per epoch and maintains a history list for later retrieval.
    """

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self._progress = RichProgressCallback()

    def log(self, metrics: dict[str, Any]) -> None:
        """Log metrics for a single epoch.

        Args:
            metrics: Dictionary of metric names → values.
        """
        self.history.append(metrics)

        # Log to Loguru
        epoch = metrics.get("epoch", "?")
        train_loss = metrics.get("train_loss", 0)
        val_loss = metrics.get("val_loss", 0)
        lr = metrics.get("lr", 0)
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.2e}"
        )

        # Display Rich table
        self._progress.log_epoch(metrics)
