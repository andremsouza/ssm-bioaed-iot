"""BioAED HPO sweep entry point — Hydra CLI."""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.hpo.optuna_sweep import run_sweep
from bioaed.utils.logging import configure_logging


def _resolve_model_key(cfg: DictConfig) -> str:
    """Extract short model name from config.

    Appends ``_pretrained`` when the model config carries a
    ``pretrained_path`` so that scratch and pretrained variants
    get separate output directories and Optuna studies.
    """
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)  # type: ignore[union-attr]
    assert isinstance(model_dict, dict)
    target = model_dict.get("_target_", "")
    key = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    if model_dict.get("pretrained_path"):
        key += "_pretrained"
    return key


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run an Optuna hyperparameter optimization sweep.

    Supports the following Hydra overrides:
        n_trials=30          Number of Optuna trials (default: 30)

    Results are saved to ``outputs/hpo/{model}/{dataset}/``.

    Args:
        cfg: Hydra-composed configuration.
    """
    configure_logging()

    n_trials = int(OmegaConf.select(cfg, "n_trials", default=30))
    model_key = _resolve_model_key(cfg)
    dataset_name = cfg.dataset.name

    output_dir = Path("outputs") / "hpo" / model_key / dataset_name
    study_name = f"{model_key}_{dataset_name}"
    logger.info(f"Starting HPO sweep: {model_key} × {dataset_name}, {n_trials} trials")

    study = run_sweep(cfg, n_trials=n_trials, output_dir=output_dir, study_name=study_name)

    logger.success(f"Sweep complete. Best trial #{study.best_trial.number}")
    logger.success(f"Best val_loss: {study.best_value:.4f}")
    logger.success(f"Best params: {study.best_params}")
    logger.success(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
