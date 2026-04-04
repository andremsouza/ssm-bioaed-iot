"""BioAED HPO sweep entry point — Hydra CLI."""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from bioaed.hpo.optuna_sweep import run_sweep
from bioaed.utils.logging import configure_logging


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run an Optuna hyperparameter optimization sweep.

    Args:
        cfg: Hydra-composed configuration.
    """
    configure_logging()
    logger.info("Starting HPO sweep")

    study = run_sweep(cfg, n_trials=50)

    logger.success(f"Sweep complete. Best trial #{study.best_trial.number}")
    logger.success(f"Best val_loss: {study.best_value:.4f}")
    logger.success(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
