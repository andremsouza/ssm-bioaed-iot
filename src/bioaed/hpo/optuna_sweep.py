"""Optuna-based hyperparameter optimization sweep."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import torch
from lightning.fabric import Fabric
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.data.datamodule import BioacousticDataModule
from bioaed.models import build_model
from bioaed.training.trainer import FabricTrainer
from bioaed.utils.reproducibility import seed_everything


def _suggest_common(trial: optuna.Trial, cfg_dict: dict[str, Any]) -> None:
    """Suggest common hyperparameters shared across all models."""
    t = cfg_dict["training"]

    t["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    t["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    t["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    cfg_dict["dataset"]["batch_size"] = t["batch_size"]
    t["scheduler"] = trial.suggest_categorical("scheduler", ["cosine", "step"])
    t["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 10)
    t["gradient_clip_val"] = trial.suggest_float("gradient_clip_val", 0.5, 5.0)

    # Class-imbalance handling
    t["use_pos_weight"] = trial.suggest_categorical("use_pos_weight", [True, False])

    # Augmentation
    aug = cfg_dict.get("augmentation", {})
    aug["enabled"] = trial.suggest_categorical("augmentation_enabled", [True, False])
    if aug["enabled"]:
        aug["freq_mask_param"] = trial.suggest_int("freq_mask_param", 4, 16)
        aug["time_mask_param"] = trial.suggest_int("time_mask_param", 10, 40)
    cfg_dict["augmentation"] = aug


def _suggest_dcqg(trial: optuna.Trial, cfg_dict: dict[str, Any]) -> None:
    """Suggest DCQG quality-gate hyperparameters."""
    qg = cfg_dict.get("quality_gate", {})
    if not qg.get("enabled", False):
        return

    qg["snr_threshold"] = trial.suggest_float("snr_threshold", -5.0, 15.0)
    qg["spectral_flatness_threshold"] = trial.suggest_float("spectral_flatness_threshold", 0.0, 0.8)
    qg["alpha"] = trial.suggest_float("alpha", 0.1, 2.0)
    qg["beta"] = trial.suggest_float("beta", 1.0, 20.0)


def _suggest_model_specific(trial: optuna.Trial, cfg_dict: dict[str, Any], model_key: str) -> None:
    """Suggest model-specific architectural hyperparameters."""
    m = cfg_dict["model"]

    if model_key == "inceptiontime":
        m["depth"] = trial.suggest_int("depth", 1, 6)
        m["n_filters"] = trial.suggest_categorical("n_filters", [16, 32, 64, 128])
        ks_choice = trial.suggest_categorical("kernel_sizes", ["small", "medium", "large"])
        ks_map = {
            "small": [5, 11, 21],
            "medium": [10, 20, 40],
            "large": [20, 40, 80],
        }
        m["kernel_sizes"] = ks_map[ks_choice]

    elif model_key == "ast":
        m["model_name"] = trial.suggest_categorical(
            "model_name", ["vit_base_patch16_224", "vit_small_patch16_224"]
        )
        m["pretrained"] = trial.suggest_categorical("pretrained", [True, False])
        # Narrower LR range for pre-trained transformers
        cfg_dict["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True
        )

    elif model_key == "audio_mamba":
        m["d_model"] = trial.suggest_categorical("d_model", [64, 128, 192, 256])
        m["n_layers"] = trial.suggest_int("n_layers", 2, 8)
        m["d_state"] = trial.suggest_categorical("d_state", [8, 16, 32, 64])
        m["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])


def _compute_pos_weight(datamodule: BioacousticDataModule) -> torch.Tensor | None:
    """Compute per-class positive weight from training label frequencies."""
    train_ds = datamodule.train_dataset
    if train_ds is None:
        return None

    # Access underlying dataset metadata through Subset
    base_ds = train_ds.dataset  # type: ignore[union-attr]
    indices = train_ds.indices  # type: ignore[union-attr]
    labels = np.array([base_ds.metadata[i]["labels"] for i in indices])

    pos_counts = labels.sum(axis=0).astype(np.float64)
    neg_counts = len(labels) - pos_counts
    # Clamp to avoid division by zero
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """Optuna objective function for a single trial.

    Suggests hyperparameters, trains a model, and returns the validation loss.

    Args:
        trial: Optuna trial object.
        cfg: Base experiment configuration.

    Returns:
        Best validation loss achieved during training.
    """
    # Create a mutable copy of the config
    trial_cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(trial_cfg, dict)

    # Determine model key
    target = trial_cfg.get("model", {}).get("_target_", "")
    model_key = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"

    # Suggest hyperparameters
    _suggest_common(trial, trial_cfg)
    _suggest_dcqg(trial, trial_cfg)
    _suggest_model_specific(trial, trial_cfg, model_key)

    # Convert back to DictConfig
    trial_omegaconf = OmegaConf.create(trial_cfg)

    # Seed
    seed_everything(trial_omegaconf.seed)

    # Setup data
    datamodule = BioacousticDataModule(trial_omegaconf)
    datamodule.setup()

    # Compute pos_weight if requested
    pos_weight = None
    if trial_cfg["training"].get("use_pos_weight", False):
        pos_weight = _compute_pos_weight(datamodule)

    # Build model
    model_cfg = OmegaConf.to_container(trial_omegaconf.model, resolve=True)
    assert isinstance(model_cfg, dict)
    target = model_cfg.pop("_target_", "")
    model_name_short = target.rsplit(".", 1)[-1] if target else "inceptiontime"
    model = build_model(model_name_short.lower(), **model_cfg)  # type: ignore[arg-type]

    # Create Fabric
    fabric = Fabric(
        accelerator="auto",
        precision=trial_omegaconf.training.precision,
    )

    # Train
    trainer = FabricTrainer(
        cfg=trial_omegaconf,
        fabric=fabric,
        output_dir=f"outputs/trial_{trial.number}",
        pos_weight=pos_weight,
    )

    best_metrics = trainer.fit(
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )

    return best_metrics.get("val_loss", float("inf"))


def run_sweep(
    cfg: DictConfig,
    n_trials: int = 50,
    n_startup_trials: int = 10,
) -> optuna.Study:
    """Run an Optuna hyperparameter sweep.

    Args:
        cfg: Base experiment configuration.
        n_trials: Number of optimization trials.
        n_startup_trials: Number of random trials before TPE kicks in.

    Returns:
        Completed Optuna study object.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study
