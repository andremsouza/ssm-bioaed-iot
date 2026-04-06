"""Ablation study runner for systematic DCQG evaluation.

Supports two evaluation modes:
    - ``multi-seed``: 10 seeds × fixed 85/15 split (default, 120 runs).
    - ``multi-seed-cv``: 10 seeds × 5-fold stratified CV (600 runs).

Designed for Hydra multirun:
    python -m bioaed.ablation --multirun \
        model=inceptiontime,ast,audio_mamba \
        dataset=aswine,anuraset \
        quality_gate.enabled=true,false \
        seed=0,1,2,3,4,5,6,7,8,9
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
    """Extract short model name from config."""
    target = OmegaConf.to_container(cfg.model, resolve=True).get("_target_", "")  # type: ignore[union-attr]
    return target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"


def _build_output_dir(cfg: DictConfig, model_key: str, fold: int | None = None) -> Path:
    """Construct the structured output directory for ablation results."""
    dataset_name = cfg.dataset.name
    dcqg_state = "dcqg_on" if cfg.quality_gate.enabled else "dcqg_off"
    base = Path("outputs") / "ablation" / model_key / dataset_name / dcqg_state / f"seed_{cfg.seed}"
    if fold is not None:
        base = base / f"fold_{fold}"
    base.mkdir(parents=True, exist_ok=True)
    return base


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
        for spectrogram, labels, _weights in test_loader:
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
    seed_everything(cfg.seed)

    model_key = _resolve_model_key(cfg)
    eval_mode = getattr(cfg, "evaluation", {})
    mode = eval_mode.get("mode", "multi-seed") if isinstance(eval_mode, dict) else "multi-seed"

    if mode == "multi-seed-cv":
        n_folds = 5
        logger.info(f"Running {n_folds}-fold CV | model={model_key} seed={cfg.seed}")

        fold_results = []
        for fold_idx in range(n_folds):
            output_dir = _build_output_dir(cfg, model_key, fold=fold_idx)
            dm = BioacousticDataModule(cfg)
            dm.setup_fold(fold_idx=fold_idx, n_folds=n_folds)

            result = _train_and_evaluate(cfg, dm, output_dir)
            result["fold"] = fold_idx
            fold_results.append(result)

            # Save per-fold result
            with open(output_dir / "results.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

        # Save aggregated fold results
        agg_dir = _build_output_dir(cfg, model_key)
        with open(agg_dir / "cv_results.json", "w") as f:
            json.dump(fold_results, f, indent=2, default=str)

        logger.success(f"CV complete: {len(fold_results)} folds saved to {agg_dir}")
    else:
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
