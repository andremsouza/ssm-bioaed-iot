"""Ablation study runner for systematic model evaluation.

Supports multi-seed evaluation with fixed 85/15 split.

Designed for Hydra multirun:
    python -m bioaed.ablation --multirun \\
        model=inceptiontime,ast,audio_mamba \\
        dataset=aswine,anuraset \\
        seed=0,1,2,3,4
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from lightning.fabric import Fabric
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.data.datamodule import BioacousticDataModule
from bioaed.evaluation.metrics import compute_metrics
from bioaed.models import _CLASS_NAME_TO_KEY, build_model
from bioaed.training.trainer import FabricTrainer
from bioaed.utils.logging import configure_logging
from bioaed.utils.reproducibility import seed_everything


def _resolve_model_key(cfg: DictConfig) -> str:
    """Extract short model name from config.

    Appends ``_pretrained`` when the model config carries a
    ``pretrained_path`` so that scratch and pretrained variants
    get separate output directories.
    """
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)  # type: ignore[union-attr]
    assert isinstance(model_dict, dict)
    target = model_dict.get("_target_", "")
    key = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    if model_dict.get("pretrained_path"):
        key += "_pretrained"
    return key


def _build_output_dir(cfg: DictConfig, model_key: str) -> Path:
    """Construct the structured output directory for ablation results."""
    output_base = OmegaConf.select(cfg, "output_base", default="ablation")
    dataset_name = cfg.dataset.name
    base = Path("outputs") / output_base / model_key / dataset_name / f"seed_{cfg.seed}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _load_hpo_params(hpo_params_dir: str, model_key: str, dataset_name: str) -> dict | None:
    """Load best_params.json from a previous HPO sweep.

    Args:
        hpo_params_dir: Root HPO output directory (e.g. ``outputs/hpo``).
        model_key: Short model name (e.g. ``inceptiontime``).
        dataset_name: Dataset name (e.g. ``aswine``).

    Returns:
        Dictionary of best hyperparameters, or None if file not found.
    """
    params_file = Path(hpo_params_dir) / model_key / dataset_name / "best_params.json"
    if not params_file.exists():
        logger.warning(f"HPO params not found: {params_file}")
        return None
    with open(params_file) as f:
        params = json.load(f)
    logger.info(f"Loaded HPO params from {params_file}: {params}")
    return params


def _apply_hpo_params(cfg: DictConfig, params: dict) -> DictConfig:
    """Merge HPO best parameters into experiment config.

    Maps flat Optuna trial param names to their nested config locations.

    Args:
        cfg: Experiment configuration.
        params: Flat dict of Optuna best parameters.

    Returns:
        Updated config with HPO parameters applied.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    # Training hyperparameters
    training_keys = {
        "learning_rate",
        "weight_decay",
        "batch_size",
        "scheduler",
        "warmup_epochs",
        "gradient_clip_val",
        "use_pos_weight",
    }
    for k, v in params.items():
        if k in training_keys:
            cfg_dict["training"][k] = v
            if k == "batch_size":
                cfg_dict["dataset"]["batch_size"] = v
        elif (
            k.startswith("freq_mask")
            or k.startswith("time_mask")
            or k.startswith("n_freq")
            or k.startswith("n_time")
        ):
            cfg_dict.setdefault("augmentation", {})[k] = v
        elif k == "augmentation_enabled":
            cfg_dict.setdefault("augmentation", {})["enabled"] = v
        else:
            # Model-specific params
            cfg_dict["model"][k] = v

    return OmegaConf.create(cfg_dict)


def _train_and_evaluate(
    cfg: DictConfig,
    datamodule: BioacousticDataModule,
    output_dir: Path,
) -> dict:
    """Run a single train+evaluate cycle and return metrics."""
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    target = model_cfg.pop("_target_", "")
    class_name = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    model_name = _CLASS_NAME_TO_KEY.get(class_name, class_name)
    model = build_model(model_name, **model_cfg)  # type: ignore[arg-type]

    fabric = Fabric(accelerator="auto", precision=cfg.training.precision)
    trainer = FabricTrainer(cfg=cfg, fabric=fabric, output_dir=str(output_dir))

    # Train
    best_val = trainer.fit(
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )

    # Load best checkpoint for test evaluation
    ckpt_path = output_dir / "checkpoint_best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])

    # Move model (and eval batches) to the same device Fabric used for training
    device = fabric.device
    model = model.to(device)

    # Evaluate on test set
    model.eval()
    all_logits, all_labels = [], []
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for spectrogram, labels in test_loader:
            spectrogram = spectrogram.to(device)
            labels = labels.to(device)
            all_logits.append(model(spectrogram))
            all_labels.append(labels)

    test_metrics = compute_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))

    return {
        "val_metrics": best_val,
        "test_metrics": test_metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run a single ablation experiment point."""
    configure_logging()

    model_key = _resolve_model_key(cfg)

    # Resume: skip combinations whose results.json already exists
    if OmegaConf.select(cfg, "resume", default=False):
        output_base = OmegaConf.select(cfg, "output_base", default="ablation")
        dataset_name = cfg.dataset.name
        candidate = (
            Path("outputs")
            / output_base
            / model_key
            / dataset_name
            / f"seed_{cfg.seed}"
            / "results.json"
        )
        if candidate.exists():
            logger.info(f"[resume] Skipping already completed run: {candidate.parent}")
            return

    seed_everything(cfg.seed)

    # Load HPO best params if available
    hpo_params_dir = OmegaConf.select(cfg, "hpo_params_dir", default=None)
    if hpo_params_dir:
        params = _load_hpo_params(hpo_params_dir, model_key, cfg.dataset.name)
        if params:
            cfg = _apply_hpo_params(cfg, params)

    # Standard multi-seed mode (single split)
    output_dir = _build_output_dir(cfg, model_key)
    dm = BioacousticDataModule(cfg)
    dm.setup()

    result = _train_and_evaluate(cfg, dm, output_dir)

    with open(output_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.success(f"Ablation run saved to {output_dir}")


if __name__ == "__main__":
    main()
