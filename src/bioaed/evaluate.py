"""BioAED evaluation entry point — Hydra CLI."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from bioaed.data.datamodule import BioacousticDataModule
from bioaed.evaluation.metrics import compute_metrics
from bioaed.evaluation.profiler import (
    count_parameters,
    estimate_macs,
    measure_throughput,
)
from bioaed.models import _CLASS_NAME_TO_KEY, build_model
from bioaed.utils.logging import configure_logging
from bioaed.utils.reproducibility import seed_everything


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run evaluation on the test set with a saved checkpoint.

    Args:
        cfg: Hydra-composed configuration.
    """
    configure_logging()
    seed_everything(cfg.seed)

    # Setup data
    datamodule = BioacousticDataModule(cfg)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg, dict)
    target = model_cfg.pop("_target_", "")
    class_name = target.rsplit(".", 1)[-1].lower() if target else "inceptiontime"
    model_name = _CLASS_NAME_TO_KEY.get(class_name, class_name)
    model = build_model(model_name, **model_cfg)  # type: ignore[arg-type]

    # Load checkpoint
    ckpt_path = Path("outputs/checkpoint_best.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning("No checkpoint found — evaluating with random weights")

    # Compute classification metrics
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for spectrogram, labels in test_loader:
            logits = model(spectrogram)
            all_logits.append(logits)
            all_labels.append(labels)

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits_cat, labels_cat)

    logger.success(f"Test metrics: {metrics}")

    # Profiling
    input_shape = (
        cfg.dataset.n_mels,
        int(cfg.dataset.segment_duration * cfg.dataset.sample_rate / cfg.dataset.hop_length) + 1,
    )
    params = count_parameters(model)
    macs = estimate_macs(model, input_shape)
    throughput = measure_throughput(model, input_shape)

    results = {**metrics, **params, **macs, **throughput}
    logger.info(f"Full results: {results}")

    # Export results
    output_path = Path("outputs/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
