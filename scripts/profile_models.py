#!/usr/bin/env python3
"""Profile all 3 model configurations for the SBBD 2026 paper.

Instantiates AST (pretrained), AudioMamba (scratch), AudioMamba (SSAMBA-pretrained)
for both datasets (AnuraSet, aSwine) and measures:
  - Parameter counts (total & trainable)
  - MACs (via ptflops)
  - Inference throughput (samples/sec) & latency (ms)

Results are saved to reports/profiling_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bioaed.evaluation.profiler import count_parameters, estimate_macs, measure_throughput

REPORTS_DIR = Path("reports")

# Dataset specs: (n_mels, time_frames, num_classes)
DATASETS = {
    "anuraset": {"n_mels": 64, "time_frames": 301, "num_classes": 42},
    "aswine": {"n_mels": 64, "time_frames": 101, "num_classes": 7},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_ast(num_classes: int, input_fdim: int, input_tdim: int):
    from bioaed.models.ast_model import AudioSpectrogramTransformer

    return AudioSpectrogramTransformer(
        num_classes=num_classes,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        fstride=10,
        tstride=10,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size="base384",
    )


def _build_mamba_scratch(num_classes: int, input_fdim: int, input_tdim: int):
    from bioaed.models.audio_mamba import AudioMamba

    return AudioMamba(
        num_classes=num_classes,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        patch_size=16,
        fstride=16,
        tstride=16,
        embed_dim=192,
        depth=4,
        d_state=16,
        bidirectional=True,
        use_cls_token=True,
    )


def _build_mamba_ssamba(num_classes: int, input_fdim: int, input_tdim: int):
    from bioaed.models.audio_mamba import AudioMamba

    return AudioMamba(
        num_classes=num_classes,
        input_fdim=input_fdim,
        input_tdim=input_tdim,
        patch_size=16,
        fstride=16,
        tstride=16,
        embed_dim=192,
        depth=24,
        d_state=16,
        bidirectional=False,
        use_cls_token=True,
        use_bimamba=True,
        use_middle_cls_token=False,
        if_divide_out=True,
        pretrained_path="checkpoints/ssamba_tiny_400.pth",
        pool_type="mean_no_cls",
    )


MODEL_BUILDERS = {
    "AST (pretrained)": _build_ast,
    "Mamba (scratch)": _build_mamba_scratch,
    "Mamba (SSAMBA)": _build_mamba_ssamba,
}


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name, builder in MODEL_BUILDERS.items():
        results[model_name] = {}
        for ds_name, ds_spec in DATASETS.items():
            print(f"\n{'=' * 60}")
            print(f"Profiling: {model_name} × {ds_name}")
            print(f"{'=' * 60}")

            model = builder(
                num_classes=ds_spec["num_classes"],
                input_fdim=ds_spec["n_mels"],
                input_tdim=ds_spec["time_frames"],
            )
            model.eval()

            input_shape = (1, ds_spec["n_mels"], ds_spec["time_frames"])

            # 1. Parameter count
            params = count_parameters(model)
            print(
                f"  Params: {params['total_params']:,} total, {params['trainable_params']:,} trainable"
            )

            # 2. MACs (Mamba requires CUDA, try CUDA first, fall back to CPU)
            try:
                macs = estimate_macs(model, input_shape, device=DEVICE)
                print(f"  MACs: {macs['estimated_macs']:,} ({macs['estimated_gflops']:.3f} GFLOPs)")
            except Exception:
                try:
                    macs = estimate_macs(model, input_shape, device="cpu")
                    print(
                        f"  MACs: {macs['estimated_macs']:,} ({macs['estimated_gflops']:.3f} GFLOPs)"
                    )
                except Exception as e:
                    print(f"  MACs: FAILED ({e})")
                    macs = {"estimated_macs": None, "estimated_gflops": None}

            # 3. Throughput
            try:
                throughput = measure_throughput(
                    model,
                    input_shape,
                    device=DEVICE,
                    num_iterations=200,
                    warmup_iterations=20,
                )
                print(f"  Throughput: {throughput['throughput_samples_per_sec']:.1f} samples/sec")
                print(f"  Latency: {throughput['avg_latency_ms']:.2f} ms")
            except Exception as e:
                print(f"  Throughput: FAILED ({e})")
                throughput = {"throughput_samples_per_sec": None, "avg_latency_ms": None}

            results[model_name][ds_name] = {
                **params,
                **macs,
                **throughput,
            }

            # Free GPU memory
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Save results
    out_path = REPORTS_DIR / "profiling_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(
        f"{'Model':<20} {'Dataset':<10} {'Params':>12} {'GFLOPs':>10} {'Throughput':>14} {'Latency':>10}"
    )
    print("-" * 80)
    for model_name, ds_results in results.items():
        for ds_name, metrics in ds_results.items():
            params_str = f"{metrics['total_params']:,}" if metrics["total_params"] else "N/A"
            gflops_str = (
                f"{metrics['estimated_gflops']:.3f}" if metrics["estimated_gflops"] else "N/A"
            )
            tp_str = (
                f"{metrics['throughput_samples_per_sec']:.1f} s/s"
                if metrics["throughput_samples_per_sec"]
                else "N/A"
            )
            lat_str = f"{metrics['avg_latency_ms']:.2f} ms" if metrics["avg_latency_ms"] else "N/A"
            print(
                f"{model_name:<20} {ds_name:<10} {params_str:>12} {gflops_str:>10} {tp_str:>14} {lat_str:>10}"
            )


if __name__ == "__main__":
    main()
