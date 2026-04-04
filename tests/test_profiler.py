"""Tests for the system profiler (parameter counting, MACs, throughput)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from bioaed.evaluation.profiler import (
    _mamba_ssm_flops_counter,
    count_parameters,
    estimate_macs,
    measure_throughput,
)
from bioaed.models.inceptiontime import InceptionTime


class TestCountParameters:
    def test_simple_model(self) -> None:
        model = nn.Linear(10, 5)
        result = count_parameters(model)
        assert result["total_params"] == 10 * 5 + 5  # weight + bias
        assert result["trainable_params"] == result["total_params"]

    def test_frozen_parameters(self) -> None:
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        result = count_parameters(model)
        assert result["total_params"] > 0
        assert result["trainable_params"] == 0


class TestEstimateMacs:
    def test_inceptiontime_macs(self) -> None:
        model = InceptionTime(num_classes=7, in_channels=64, depth=1, n_filters=16)
        result = estimate_macs(model, input_shape=(64, 100))
        assert result["estimated_macs"] > 0
        assert result["estimated_gflops"] > 0

    def test_linear_model_macs(self) -> None:
        model = nn.Sequential(nn.Flatten(), nn.Linear(64 * 100, 10))
        result = estimate_macs(model, input_shape=(64, 100))
        assert result["estimated_macs"] > 0


class TestMeasureThroughput:
    def test_throughput_positive(self) -> None:
        model = InceptionTime(num_classes=7, in_channels=64, depth=1, n_filters=16)
        result = measure_throughput(
            model, input_shape=(64, 100), num_iterations=5, warmup_iterations=1
        )
        assert result["throughput_samples_per_sec"] > 0
        assert result["avg_latency_ms"] > 0


class TestMambaFlopsCounter:
    def test_counts_flops(self) -> None:
        module = MagicMock()
        module.d_model = 64
        module.d_state = 16
        module.__flops__ = 0
        # Input shape: (batch=2, seq_len=100, d_model=64)
        inp = torch.randn(2, 100, 64)
        _mamba_ssm_flops_counter(module, (inp,), None)
        # Expected: 2 * 2 * 100 * 64 * 16 = 409600
        assert module.__flops__ == 2 * 2 * 100 * 64 * 16

    def test_skips_without_attrs(self) -> None:
        module = MagicMock(spec=[])  # no d_model or d_state
        _mamba_ssm_flops_counter(module, (torch.randn(1, 10),), None)
        # Should not raise

    def test_2d_input(self) -> None:
        module = MagicMock()
        module.d_model = 32
        module.d_state = 8
        module.__flops__ = 0
        inp = torch.randn(4, 50)
        _mamba_ssm_flops_counter(module, (inp,), None)
        assert module.__flops__ > 0

    def test_1d_input(self) -> None:
        module = MagicMock()
        module.d_model = 16
        module.d_state = 4
        module.__flops__ = 0
        inp = torch.randn(8)  # 1D
        _mamba_ssm_flops_counter(module, (inp,), None)
        # Falls back to seq_len=1
        assert module.__flops__ == 2 * 8 * 1 * 16 * 4
