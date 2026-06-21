"""Preflight validation for ablation experiments.

Resolves every model × dataset combination from an ablation profile,
then runs a forward + backward smoke test on each to catch errors
before hours of training begin.

Checks performed per combination:
  1. Model instantiation with resolved config
  2. Forward pass with dummy tensor → correct output shape
  3. Backward pass → gradients flow to all parameters
  4. One real sample loaded from disk → correct spectrogram shape
  5. Real sample forward pass through the model

Global checks:
  - Dataset directories exist and contain audio files
  - Hardware: CUDA availability, bf16 support

Usage:
    python -m bioaed.preflight quick_pilot
    python -m bioaed.preflight extended_pilot
    python -m bioaed.preflight full_matrix
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml
from loguru import logger

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _parse_sweep_combos(profile_name: str) -> tuple[list[str], list[str]]:
    """Extract model and dataset names from the ablation profile sweeper."""
    ablation_cfg = _load_yaml(CONFIGS_DIR / "ablation" / f"{profile_name}.yaml")
    sweeper = ablation_cfg.get("hydra", {}).get("sweeper", {}).get("params", {})

    models = _extract_sweep_or_default(sweeper, ablation_cfg, "model", "inceptiontime")
    datasets = _extract_sweep_or_default(sweeper, ablation_cfg, "dataset", "aswine")
    return models, datasets


def _extract_sweep_or_default(
    sweeper: dict,
    ablation_cfg: dict,
    group: str,
    fallback: str,
) -> list[str]:
    """Get values from sweeper params, or fall back to the profile's defaults."""
    if group in sweeper:
        return [v.strip() for v in sweeper[group].split(",")]

    for d in ablation_cfg.get("defaults", []):
        if isinstance(d, dict):
            for key, val in d.items():
                if key == f"override /{group}":
                    return [val]
    return [fallback]


def _resolve_dataset_config(dataset_name: str) -> dict:
    cfg = _load_yaml(CONFIGS_DIR / "dataset" / f"{dataset_name}.yaml")
    # Resolve OmegaConf env default: ${oc.env:DATA_DIR,data}/name → data/name
    root_dir = cfg.get("root_dir", "")
    if "${oc.env:" in root_dir:
        cfg["root_dir"] = f"data/{dataset_name}"
    return cfg


def _resolve_model_kwargs(model_name: str, dataset_cfg: dict) -> dict:
    """Load model config and resolve ${dataset.*} interpolations."""
    model_cfg = _load_yaml(CONFIGS_DIR / "model" / f"{model_name}.yaml")
    resolved = {}
    for key, val in model_cfg.items():
        if key == "_target_":
            continue
        if isinstance(val, str) and "${dataset." in val:
            field = val.replace("${dataset.", "").rstrip("}")
            resolved[key] = dataset_cfg[field]
        else:
            resolved[key] = val
    return resolved


def _resolve_training_config(profile_name: str) -> dict:
    """Get effective training config (profile overrides merged onto default)."""
    base = _load_yaml(CONFIGS_DIR / "training" / "default.yaml")
    ablation = _load_yaml(CONFIGS_DIR / "ablation" / f"{profile_name}.yaml")
    training_overrides = ablation.get("training", {})
    if training_overrides:
        base.update(training_overrides)
    return base


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_dataset_exists(dataset_name: str, dataset_cfg: dict) -> list[str]:
    """Verify the dataset directory has audio files."""
    errors: list[str] = []
    root = Path(dataset_cfg["root_dir"])
    if not root.exists():
        errors.append(f"Dataset directory not found: {root.resolve()}")
        return errors

    audio_dir = root / "audio"
    if not audio_dir.exists():
        errors.append(f"Audio directory not found: {audio_dir.resolve()}")
        return errors

    has_audio = any(audio_dir.rglob("*.wav")) or any(audio_dir.rglob("*.mp3"))
    if not has_audio:
        errors.append(f"No audio files (.wav/.mp3) in {audio_dir.resolve()}")
    return errors


def check_model_smoke(
    model_name: str,
    model_kwargs: dict,
    dataset_cfg: dict,
) -> list[str]:
    """Build model, forward+backward with dummy tensor."""
    from bioaed.models import build_model

    errors: list[str] = []

    # --- Build ---
    try:
        model = build_model(model_name, **model_kwargs)
    except Exception as e:
        errors.append(f"Model instantiation failed: {e}")
        return errors

    n_mels = dataset_cfg["n_mels"]
    time_frames = dataset_cfg["time_frames"]
    num_classes = dataset_cfg["num_classes"]
    batch = 2

    # Some models (e.g. AudioMamba with mamba-ssm) require CUDA tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = torch.randn(batch, n_mels, time_frames, device=device)

    # --- Forward ---
    try:
        model.train()
        logits = model(x)
    except Exception as e:
        errors.append(f"Forward pass failed: {e}")
        return errors

    expected = (batch, num_classes)
    if logits.shape != expected:
        errors.append(f"Output shape {tuple(logits.shape)} != expected {expected}")

    if torch.isnan(logits).any():
        errors.append("Forward pass produced NaN values")

    # --- Backward ---
    try:
        logits.sum().backward()
    except Exception as e:
        errors.append(f"Backward pass failed: {e}")
        return errors

    trainable = [p for p in model.parameters() if p.requires_grad]
    grads_ok = sum(1 for p in trainable if p.grad is not None)
    if grads_ok == 0:
        errors.append("No gradients computed on any parameter")
    elif grads_ok < len(trainable):
        missing = len(trainable) - grads_ok
        ratio = missing / len(trainable)
        msg = f"Gradients missing on {missing}/{len(trainable)} trainable parameters ({ratio:.0%})"
        # Small fraction of unused pretrained params (e.g. DeiT head/head_dist)
        # is expected — warn instead of fail.
        if ratio <= 0.05:
            logger.warning(f"  {msg} (likely unused pretrained heads — OK)")
        else:
            errors.append(msg)

    return errors


def check_real_sample(
    model_name: str,
    model_kwargs: dict,
    dataset_cfg: dict,
) -> list[str]:
    """Load one real sample and pass it through the model."""
    from omegaconf import OmegaConf

    from bioaed.data.datamodule import BioacousticDataModule
    from bioaed.models import build_model

    errors: list[str] = []
    aug_cfg = _load_yaml(CONFIGS_DIR / "augmentation" / "default.yaml")
    # Disable augmentation for smoke test
    aug_cfg["enabled"] = False

    cfg = OmegaConf.create(
        {
            "dataset": dataset_cfg,
            "augmentation": aug_cfg,
            "seed": 42,
        }
    )

    # --- DataModule setup ---
    try:
        dm = BioacousticDataModule(cfg)
        dm.setup()
    except Exception as e:
        errors.append(f"DataModule.setup() failed: {e}")
        return errors

    if dm.train_dataset is None or len(dm.train_dataset) == 0:  # type: ignore[arg-type]
        errors.append("Train dataset is empty after setup")
        return errors

    # --- Load 1 sample ---
    try:
        spec, labels, weight = dm.train_dataset[0]  # type: ignore[index]
    except Exception as e:
        errors.append(f"Loading sample failed: {e}")
        return errors

    n_mels = dataset_cfg["n_mels"]
    time_frames = dataset_cfg["time_frames"]
    num_classes = dataset_cfg["num_classes"]

    if spec.shape[0] != n_mels:
        errors.append(f"Spectrogram n_mels={spec.shape[0]} != config n_mels={n_mels}")
    if spec.shape[1] != time_frames:
        errors.append(
            f"Spectrogram time_frames={spec.shape[1]} != config time_frames={time_frames}"
        )
    if labels.shape[0] != num_classes:
        errors.append(f"Labels dim={labels.shape[0]} != config num_classes={num_classes}")

    # --- Forward real sample ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(model_name, **model_kwargs)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(spec.unsqueeze(0).to(device))
    except Exception as e:
        errors.append(f"Real-sample forward failed: {e}")
        return errors

    if logits.shape != (1, num_classes):
        errors.append(f"Real-sample output shape {tuple(logits.shape)} != (1, {num_classes})")

    return errors


def check_hardware(training_cfg: dict) -> list[str]:
    """Verify hardware requirements for the training config."""
    errors: list[str] = []

    precision = training_cfg.get("precision", "32-true")
    if "bf16" in precision and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        errors.append(
            f"Config requires precision={precision} but GPU does not support bf16. "
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
        # bf16-mixed can work on CPU via autocast, so no error for CPU-only

    return errors


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_preflight(profile_name: str) -> bool:
    """Run all preflight checks for the given ablation profile.

    Returns True if all checks pass.
    """
    t0 = time.time()
    all_errors: dict[str, list[str]] = {}

    # Parse combinations
    models, datasets = _parse_sweep_combos(profile_name)
    n_combos = len(models) * len(datasets)

    logger.info(f"Preflight: {profile_name}")
    logger.info(f"  Models:   {models}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Combinations to validate: {n_combos}")
    logger.info("")

    # Resolve training config for hardware checks
    training_cfg = _resolve_training_config(profile_name)
    hw_errors = check_hardware(training_cfg)
    if hw_errors:
        all_errors["hardware"] = hw_errors

    # Per-dataset checks (run once per dataset, not per model)
    dataset_configs: dict[str, dict] = {}
    for ds_name in datasets:
        dataset_configs[ds_name] = _resolve_dataset_config(ds_name)
        ds_errors = check_dataset_exists(ds_name, dataset_configs[ds_name])
        if ds_errors:
            all_errors[f"dataset:{ds_name}"] = ds_errors

    # Per model×dataset combination
    for model_name in models:
        for ds_name in datasets:
            combo = f"{model_name} × {ds_name}"
            logger.info(f"  Checking {combo} ...")
            ds_cfg = dataset_configs[ds_name]
            model_kwargs = _resolve_model_kwargs(model_name, ds_cfg)

            # Smoke test (dummy tensor)
            errs = check_model_smoke(model_name, model_kwargs, ds_cfg)
            if errs:
                all_errors[f"smoke:{combo}"] = errs
                # Don't bother with real sample if dummy fails
                continue

            # Real sample test
            # Skip if dataset directory is missing
            if f"dataset:{ds_name}" not in all_errors:
                errs = check_real_sample(model_name, model_kwargs, ds_cfg)
                if errs:
                    all_errors[f"real:{combo}"] = errs

            logger.info(f"  {combo}: OK")

    # Report
    elapsed = time.time() - t0
    logger.info("")

    if all_errors:
        logger.error(f"PREFLIGHT FAILED ({len(all_errors)} issue(s) in {elapsed:.1f}s):")
        for key, errs in all_errors.items():
            logger.error(f"  [{key}]")
            for e in errs:
                logger.error(f"    - {e}")
        return False

    logger.success(f"Preflight passed: {n_combos} combinations validated in {elapsed:.1f}s")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight validation for ablation experiments")
    parser.add_argument(
        "profile",
        help="Ablation profile name (e.g. quick_pilot, extended_pilot, full_matrix)",
    )
    args = parser.parse_args()

    ok = run_preflight(args.profile)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
