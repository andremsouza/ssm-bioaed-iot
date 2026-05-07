"""System profiler: parameter count, MACs, and inference throughput."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from ptflops import get_model_complexity_info


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dict with ``total_params`` and ``trainable_params``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def _mamba_ssm_flops_counter(module: nn.Module, input: Any, output: Any) -> None:
    """Custom FLOPs handler for Mamba SSM blocks.

    Estimates MACs from the SSM state expansion:
        MACs ≈ B * L * (D * N + D * N) = 2 * B * L * D * N

    Symbols:
        - B: batch size
        - L: sequence length (number of time-frequency tokens)
        - D: d_model (inner state/channel dimension of the SSM)
        - N: d_state (SSM state dimension controlling memory capacity)

    The 2× factor covers two sequential passes per token:
        1. Discretisation step (Δ, A, B computation) — projects continuous parameters
           to discrete recurrence matrices.
        2. Selective scan recurrence — propagates the hidden state across L tokens.
    """
    if not hasattr(module, "d_model") or not hasattr(module, "d_state"):
        return
    d_model = module.d_model
    d_state = module.d_state
    # Estimate sequence length from input
    inp = input[0] if isinstance(input, tuple) else input
    if inp.dim() >= 2:
        seq_len = inp.shape[1] if inp.dim() == 3 else inp.shape[-1]
    else:
        seq_len = 1
    batch_size = inp.shape[0]
    # SSM selective scan: ~2 * B * L * D * N (discretize + scan)
    macs = 2 * batch_size * seq_len * d_model * d_state
    module.__flops__ += int(macs)


def estimate_macs(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
) -> dict[str, Any]:
    """Estimate MACs using ``ptflops`` for accurate counting across all layers.

    Handles Conv1d, Conv2d, Linear, attention layers, and optionally Mamba
    SSM blocks via a custom handler.

    Args:
        model: PyTorch model.
        input_shape: Shape of a single input tensor (excluding batch dim).
        device: Device for the estimation.

    Returns:
        Dict with ``estimated_macs`` and ``estimated_gflops``.
    """
    # Build custom hooks for Mamba blocks if present
    custom_hooks: dict[type, Any] = {}
    try:
        from mamba_ssm.modules.mamba_simple import Mamba

        custom_hooks[Mamba] = _mamba_ssm_flops_counter
    except ImportError:
        pass

    model = model.to(device)
    macs, _ = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        custom_modules_hooks=custom_hooks,
    )

    return {
        "estimated_macs": int(macs),
        "estimated_gflops": macs / 1e9,
    }


def measure_throughput(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cpu",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure inference throughput in samples per second.

    Args:
        model: PyTorch model.
        input_shape: Shape of a single input tensor (excluding batch dim).
        device: Device for inference.
        num_iterations: Number of timed forward passes.
        warmup_iterations: Warmup iterations (not timed).
        batch_size: Number of samples per forward pass.

    Returns:
        Dict with ``throughput_samples_per_sec`` and ``avg_latency_ms``.
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / num_iterations) * 1000
    throughput = (num_iterations * batch_size) / elapsed

    logger.info(
        f"Throughput (batch={batch_size}): {throughput:.1f} samples/sec"
        f" | Latency: {avg_latency_ms:.2f} ms"
    )

    return {
        "throughput_samples_per_sec": throughput,
        "avg_latency_ms": avg_latency_ms,
    }


def measure_peak_memory_mb(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: str = "cuda",
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure peak GPU memory during a forward pass.

    Args:
        model: PyTorch model.
        input_shape: Shape of a single input tensor (excluding batch dim).
        device: Device for inference (must be ``cuda`` for meaningful results).
        batch_size: Number of samples per forward pass.

    Returns:
        Dict with ``peak_memory_mb``.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        logger.warning("Peak memory measurement requires a CUDA device; returning 0.")
        return {"peak_memory_mb": 0.0}

    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, *input_shape, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024**2)
    logger.info(f"Peak GPU memory (batch={batch_size}): {peak_mb:.1f} MB")
    return {"peak_memory_mb": peak_mb}
