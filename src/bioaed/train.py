"""BioAED training entry point — Hydra CLI."""

from __future__ import annotations

import hydra
from lightning.fabric import Fabric
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.data.datamodule import BioacousticDataModule
from bioaed.models import build_model
from bioaed.training.trainer import FabricTrainer
from bioaed.utils.config_schemas import ExperimentConfig
from bioaed.utils.logging import configure_logging, log_config
from bioaed.utils.reproducibility import seed_everything


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run a training experiment.

    This function is the main Hydra entry point. It validates the config,
    seeds all RNGs, sets up data / model / trainer, and runs the training loop.

    Args:
        cfg: Hydra-composed configuration.
    """
    configure_logging()

    # Validate config with Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    validated = ExperimentConfig(**cfg_dict)  # type: ignore[arg-type]
    logger.info(f"Experiment: {validated.experiment_name}")
    log_config(cfg_dict)  # type: ignore[arg-type]

    # Seed everything
    seed_everything(cfg.seed)

    # Setup data
    datamodule = BioacousticDataModule(cfg)
    datamodule.setup()

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    target = model_cfg.pop("_target_", "")
    model_name = target.rsplit(".", 1)[-1] if target else "inceptiontime"
    model = build_model(model_name.lower(), **model_cfg)  # type: ignore[arg-type]

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name} | Parameters: {param_count:,}")

    # Create Fabric
    fabric = Fabric(
        accelerator="auto",
        precision=cfg.training.precision,
    )

    # Train
    trainer = FabricTrainer(cfg=cfg, fabric=fabric)
    best_metrics = trainer.fit(
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
    )

    logger.success(f"Training complete. Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()
