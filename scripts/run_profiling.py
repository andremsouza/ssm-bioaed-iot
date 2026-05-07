"""Run extended profiling (batch=1 and batch=128, peak GPU memory).

Produces reports/profiling_results_extended.json with keys::

    {model_name: {dataset: {batch_1: {...}, batch_128: {...}}}}

Uses saved best checkpoints from outputs/ablation/ where available;
falls back to random weights (profiling does not require trained weights).

Usage::

    python scripts/run_profiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Ensure project source is importable from any cwd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bioaed.evaluation.profiler import (
    count_parameters,
    estimate_macs,
    measure_peak_memory_mb,
    measure_throughput,
)
from bioaed.models import build_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [1, 128]

MODELS = {
    "AST (pretrained)": {
        "key": "ast",
        "ablation_key": "audiospectrogramtransformer",
        "build_kwargs": {
            "input_fdim": 64,
            "input_tdim": 301,
            "num_classes": 42,
        },
    },
    "Mamba (SSAMBA)": {
        "key": "audio_mamba",
        "ablation_key": "audiomamba_pretrained",
        "build_kwargs": {
            "input_fdim": 64,
            "input_tdim": 301,
            "num_classes": 42,
            "embed_dim": 192,
            "depth": 24,
            "d_state": 16,
            "bidirectional": False,
            "use_cls_token": True,
            "use_bimamba": True,
            "use_middle_cls_token": False,
            "if_divide_out": True,
            "pool_type": "mean_no_cls",
            "pretrained_path": None,  # skip pretrained load; random weights are fine for profiling
        },
    },
    "Mamba (scratch)": {
        "key": "audio_mamba",
        "ablation_key": "audiomamba",
        "build_kwargs": {
            "input_fdim": 64,
            "input_tdim": 301,
            "num_classes": 42,
            "embed_dim": 192,
            "depth": 4,
            "d_state": 16,
            "bidirectional": True,
            "use_cls_token": True,
        },
    },
}

DATASETS = {
    "anuraset": {"input_shape": (64, 301), "num_classes": 42},
    "aswine": {"input_shape": (64, 101), "num_classes": 7},
}

ABLATION_ROOT = ROOT / "outputs" / "ablation"


def load_checkpoint(model: torch.nn.Module, model_key: str, dataset: str) -> bool:
    """Try to load best checkpoint; return True on success."""
    ckpt = ABLATION_ROOT / model_key / dataset / "seed_0" / "checkpoint_best.pt"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state_dict"])
            return True
        except Exception as e:
            print(f"  [warn] checkpoint load failed ({e}); using random weights")
    return False


def main() -> None:
    results: dict = {}

    for model_name, model_cfg in MODELS.items():
        results[model_name] = {}
        model_key = model_cfg["key"]
        ablation_key = model_cfg["ablation_key"]
        for dataset_name, dataset_cfg in DATASETS.items():
            print(f"\n=== {model_name} / {dataset_name} ===")
            input_shape = dataset_cfg["input_shape"]
            num_classes = dataset_cfg["num_classes"]

            # Build model with correct num_classes for dataset
            kwargs = dict(model_cfg["build_kwargs"])
            kwargs["num_classes"] = num_classes
            # Adjust input dims for aSwine
            if dataset_name == "aswine":
                kwargs["input_fdim"] = 64
                kwargs["input_tdim"] = 101

            model = build_model(model_key, **kwargs)
            load_checkpoint(model, ablation_key, dataset_name)
            model = model.to(DEVICE)
            model.eval()

            dataset_results: dict = {}

            # Static metrics (only need batch=1)
            params = count_parameters(model)
            macs = estimate_macs(model, input_shape, device=DEVICE)
            print(f"  Params: {params['total_params']:,}  GFLOPs: {macs['estimated_gflops']:.4f}")

            for bs in BATCH_SIZES:
                print(f"  Profiling batch={bs} ...")
                thr = measure_throughput(
                    model,
                    input_shape,
                    device=DEVICE,
                    batch_size=bs,
                    num_iterations=100,
                    warmup_iterations=10,
                )
                mem = measure_peak_memory_mb(
                    model,
                    input_shape,
                    device=DEVICE,
                    batch_size=bs,
                )
                dataset_results[f"batch_{bs}"] = {
                    **params,
                    **macs,
                    **thr,
                    **mem,
                }

            results[model_name][dataset_name] = dataset_results

    out = ROOT / "reports" / "profiling_results_extended.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
