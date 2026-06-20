"""Optuna-based hyperparameter optimization sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from lightning.fabric import Fabric
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.data.datamodule import BioacousticDataModule
from bioaed.models import _CLASS_NAME_TO_KEY, build_model
from bioaed.training.trainer import FabricTrainer
from bioaed.utils.reproducibility import seed_everything


def _suggest_common(
    trial: optuna.Trial,
    cfg_dict: dict[str, Any],
    skip_lr: bool = False,
) -> None:
    """Suggest common hyperparameters shared across all models.

    Args:
        trial: Optuna trial object.
        cfg_dict: Mutable config dict to update in-place.
        skip_lr: When ``True``, skip the ``learning_rate`` suggestion so that
            the model-specific function can register it with a narrower range.
            Required for pretrained models (AST, SSAMBA) to prevent Optuna
            from caching the wide [1e-5, 1e-2] value before the narrow range
            is suggested, which would silently ignore the narrower bounds.
    """
    t = cfg_dict["training"]

    if not skip_lr:
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


def _suggest_model_specific(trial: optuna.Trial, cfg_dict: dict[str, Any], model_key: str) -> None:
    """Suggest model-specific architectural hyperparameters.

    For pretrained models (AST, SSAMBA) the backbone architecture is fixed by
    the checkpoint, so only training hyperparameters are tuned.  Critically,
    the learning-rate range is narrowed to [1e-5, 5e-4] (versus [1e-5, 1e-2]
    for scratch models) to avoid catastrophic forgetting: large learning rates
    would irreversibly overwrite the rich representations acquired during
    AudioSet pre-training in just a few gradient steps.
    """
    m = cfg_dict["model"]

    if model_key == "inceptiontime":
        m["depth"] = trial.suggest_int("depth", 3, 9)
        m["n_filters"] = trial.suggest_categorical("n_filters", [16, 32, 64])
        m["kernel_size"] = trial.suggest_categorical("kernel_size", [21, 41, 81])
        m["use_bottleneck"] = trial.suggest_categorical("use_bottleneck", [True, False])
        m["bottleneck_size"] = trial.suggest_categorical("bottleneck_size", [16, 32, 64])

    elif model_key == "ast":
        m["model_size"] = trial.suggest_categorical("model_size", ["base384", "base224"])
        m["imagenet_pretrain"] = True
        m["audioset_pretrain"] = True
        # Narrower LR range for pre-trained transformers
        cfg_dict["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True
        )

    elif model_key == "audio_mamba":
        # Scratch AudioMamba — tune architecture
        m["embed_dim"] = trial.suggest_categorical("embed_dim", [64, 128, 192, 256])
        m["depth"] = trial.suggest_categorical("depth", [2, 4, 6, 8])
        m["d_state"] = trial.suggest_categorical("d_state", [8, 16, 32, 64])
        m["patch_size"] = trial.suggest_categorical("patch_size", [8, 16, 32])

    elif model_key == "audio_mamba_pretrained":
        # Pretrained SSAMBA — architecture is frozen by checkpoint.
        # Tune fine-tuning-specific hyperparameters only.
        cfg_dict["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True
        )
        m["pool_type"] = trial.suggest_categorical("pool_type", ["mean_no_cls", "cls", "mean"])
        m["drop_path_rate"] = trial.suggest_float("drop_path_rate", 0.0, 0.3)


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

    # Determine model key (with _pretrained suffix when applicable)
    model_dict = trial_cfg.get("model", {})
    target = model_dict.get("_target_", "")
    class_name = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    model_key = _CLASS_NAME_TO_KEY.get(class_name, class_name)
    if model_dict.get("pretrained_path"):
        model_key += "_pretrained"

    # Pretrained models (AST, SSAMBA) use a narrower LR range tuned in
    # _suggest_model_specific.  Skip the wide LR registration in
    # _suggest_common so the first suggest_float call for "learning_rate"
    # uses the correct narrow distribution (Optuna caches on first call).
    skip_lr = model_key in ("ast", "audio_mamba_pretrained")

    # Suggest hyperparameters
    _suggest_common(trial, trial_cfg, skip_lr=skip_lr)
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
    class_name = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    model_name_short = _CLASS_NAME_TO_KEY.get(class_name, class_name)
    model = build_model(model_name_short, **model_cfg)  # type: ignore[arg-type]

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

    val_loss = best_metrics.get("val_loss", float("inf"))
    # Guard against NaN/Inf val_loss (e.g. exploding gradients at high LR)
    if not isinstance(val_loss, float) or val_loss != val_loss:  # NaN check
        logger.warning(f"Trial {trial.number}: non-finite val_loss={val_loss}; returning inf")
        return float("inf")
    return val_loss


def run_sweep(
    cfg: DictConfig,
    n_trials: int = 50,
    n_startup_trials: int = 10,
    output_dir: str | Path | None = None,
    study_name: str | None = None,
) -> optuna.Study:
    """Run an Optuna hyperparameter sweep.

    Uses SQLite-backed storage when *output_dir* is provided so that
    partial progress survives interruptions.  On re-launch the study is
    loaded automatically and only the remaining trials are executed.

    Args:
        cfg: Base experiment configuration.
        n_trials: Number of optimization trials.
        n_startup_trials: Number of random trials before TPE kicks in.
        output_dir: Directory to save ``best_params.json`` and the
            SQLite study database.  Created automatically when given.
        study_name: Optuna study name used for persistent storage.
            Defaults to ``"hpo"`` when *output_dir* is provided.

    Returns:
        Completed Optuna study object.
    """
    storage: str | None = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{out.resolve()}/study.db"

    study = optuna.create_study(
        study_name=study_name or ("hpo" if storage else None),
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed, n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # Heal any trials left in RUNNING state by a previous crashed process.
    if storage is not None and storage.startswith("sqlite:///"):
        import sqlite3 as _sqlite3

        db_path = storage.removeprefix("sqlite:///")
        try:
            with _sqlite3.connect(db_path, timeout=10) as _conn:
                _conn.isolation_level = None
                _conn.execute("UPDATE trials SET state = 'FAIL' WHERE state = 'RUNNING'")
        except Exception as _e:
            logger.warning(f"Could not heal stale RUNNING trials: {_e}")

    # Count completed trials directly from storage (avoids stale in-process cache).
    if storage is not None and storage.startswith("sqlite:///"):
        import sqlite3 as _sqlite3

        db_path = storage.removeprefix("sqlite:///")
        with _sqlite3.connect(db_path, timeout=10) as _conn:
            _row = _conn.execute("SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'").fetchone()
            completed = _row[0] if _row else 0
    else:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    # Callback: persist best_params after every successful trial so a crash
    # on a later trial does not discard already-found optima.
    def _checkpoint_callback(st: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        if output_dir is not None and _trial.state == optuna.trial.TrialState.COMPLETE:
            _out = Path(output_dir)
            with open(_out / "best_params.json", "w") as _f:
                json.dump(st.best_params, _f, indent=2)

    if remaining == 0:
        logger.info(f"All {n_trials} trials already complete — skipping optimization")
    else:
        if completed > 0:
            logger.info(
                f"Resuming study: {completed}/{n_trials} trials done, {remaining} remaining"
            )
        study.optimize(
            lambda trial: objective(trial, cfg),
            n_trials=remaining,
            show_progress_bar=True,
            callbacks=[_checkpoint_callback],
        )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Persist best params
    if output_dir is not None:
        out = Path(output_dir)
        with open(out / "best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)
        logger.info(f"HPO artifacts saved to {out}")

    return study
