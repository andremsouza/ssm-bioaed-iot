"""Lightning Fabric-based trainer with quality-weighted BCE loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from lightning.fabric import Fabric
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bioaed.training.callbacks import EarlyStopping, MetricLogger, RichProgressCallback
from bioaed.training.optimizer import build_optimizer_and_scheduler


class FabricTrainer:
    """Custom training loop built on Lightning Fabric.

    Retains full control over the gradient update / loss computation while
    leveraging Fabric for device placement, mixed precision, and distributed
    execution.

    Args:
        cfg: OmegaConf DictConfig with ``training`` group.
        fabric: Pre-configured Lightning Fabric instance.
        output_dir: Directory for saving checkpoints and logs.
    """

    def __init__(
        self,
        cfg: DictConfig,
        fabric: Fabric,
        output_dir: str | Path = "outputs",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.fabric = fabric
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pos_weight = pos_weight

        # Callbacks
        self.early_stopping = EarlyStopping(patience=cfg.training.patience)
        self.progress = RichProgressCallback()
        self.metric_logger = MetricLogger()

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
    ) -> dict[str, float]:
        """Run the full training loop.

        Args:
            model: The neural network to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary with best validation metrics.
        """
        # Build optimizer & scheduler
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            learning_rate=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            max_epochs=self.cfg.training.max_epochs,
            scheduler_name=self.cfg.training.scheduler,
            warmup_epochs=self.cfg.training.warmup_epochs,
        )

        # Optional torch.compile
        if getattr(self.cfg.training, "compile", False):
            model = torch.compile(model)  # type: ignore[assignment]
            logger.info("Model compiled with torch.compile()")

        # Setup with Fabric
        model, optimizer = self.fabric.setup(model, optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)  # type: ignore[assignment]
        val_loader = self.fabric.setup_dataloaders(val_loader)  # type: ignore[assignment]

        # Loss function (BCEWithLogitsLoss for multi-label)
        pw = self.pos_weight
        if pw is not None:
            pw = self.fabric.to_device(pw)
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

        best_val_loss = float("inf")
        best_metrics: dict[str, float] = {}

        for epoch in range(self.cfg.training.max_epochs):
            # === Training phase ===
            model.train()
            train_loss = 0.0
            num_batches = 0

            for batch_idx, (spectrogram, labels, quality_weights) in enumerate(train_loader):
                # Forward pass
                logits = model(spectrogram)
                loss_unreduced = criterion(logits, labels)

                # Apply quality-gate weights (per-sample weighting)
                weighted_loss = loss_unreduced * quality_weights.unsqueeze(-1)
                loss = weighted_loss.mean()

                # Gradient accumulation
                is_accumulating = (batch_idx + 1) % self.cfg.training.accumulate_grad_batches != 0
                self.fabric.backward(loss)

                if not is_accumulating:
                    # Gradient clipping
                    self.fabric.clip_gradients(
                        model, optimizer, max_norm=self.cfg.training.gradient_clip_val
                    )
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / max(num_batches, 1)

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            # === Validation phase ===
            val_loss = self._validate(model, val_loader, criterion)

            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            self.metric_logger.log(metrics)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics.copy()
                self._save_checkpoint(model, optimizer, epoch, "best")
                logger.info(f"New best model at epoch {epoch + 1} (val_loss={val_loss:.4f})")

            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Save final checkpoint
        self._save_checkpoint(model, optimizer, epoch, "last")

        return best_metrics

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader[Any],
        criterion: nn.Module,
    ) -> float:
        """Run validation and return average loss.

        Args:
            model: The model to evaluate.
            val_loader: Validation DataLoader.
            criterion: Loss function.

        Returns:
            Average validation loss.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0

        for spectrogram, labels, _quality_weights in val_loader:
            logits = model(spectrogram)
            loss_unreduced = criterion(logits, labels)
            # Unweighted val loss prevents biased model selection from DCQG
            total_loss += loss_unreduced.mean().item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        tag: str,
    ) -> None:
        """Save a training checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch number.
            tag: Checkpoint tag (e.g., ``"best"`` or ``"last"``).
        """
        ckpt_path = self.output_dir / f"checkpoint_{tag}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        self.fabric.save(ckpt_path, state)
        logger.debug(f"Saved checkpoint: {ckpt_path}")
