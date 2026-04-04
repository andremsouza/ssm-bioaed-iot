"""Tests for audio transforms: LogMelSpectrogram, ZNormalize, SpecAugment."""

from __future__ import annotations

import torch

from bioaed.features.transforms import LogMelSpectrogram, SpecAugment, ZNormalize


class TestLogMelSpectrogram:
    def test_output_shape(self) -> None:
        transform = LogMelSpectrogram(sample_rate=16000, n_mels=64, hop_length=160, win_length=400)
        waveform = torch.randn(1, 1, 16000)
        out = transform(waveform)
        # (batch, channels, n_mels, time)
        assert out.shape[2] == 64  # n_mels

    def test_log_scale(self) -> None:
        transform = LogMelSpectrogram(sample_rate=16000, n_mels=64, hop_length=160, win_length=400)
        waveform = torch.randn(1, 1, 16000)
        out = transform(waveform)
        # Log of small positive values → all values should be finite
        assert torch.isfinite(out).all()


class TestZNormalize:
    def test_identity_with_defaults(self) -> None:
        znorm = ZNormalize()
        x = torch.randn(64, 100)
        out = znorm(x)
        # Default mean=0, std=1, so output ≈ input
        assert torch.allclose(out, x, atol=1e-5)

    def test_normalization(self) -> None:
        mean = torch.ones(64, 1) * 5.0
        std = torch.ones(64, 1) * 2.0
        znorm = ZNormalize(mean=mean, std=std)
        x = torch.ones(64, 100) * 5.0
        out = znorm(x)
        # (5 - 5) / (2 + 1e-9) ≈ 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)

    def test_from_dataset_stats(self) -> None:
        mean = torch.randn(64, 1)
        std = torch.rand(64, 1) + 0.1
        znorm = ZNormalize.from_dataset_stats(mean=mean, std=std)
        x = torch.randn(64, 100)
        out = znorm(x)
        assert out.shape == x.shape

    def test_output_shape_preserved(self) -> None:
        mean = torch.zeros(64, 1)
        std = torch.ones(64, 1)
        znorm = ZNormalize(mean=mean, std=std)
        x = torch.randn(2, 64, 100)  # batch dimension
        out = znorm(x)
        assert out.shape == x.shape


class TestSpecAugment:
    def test_output_shape_preserved(self) -> None:
        aug = SpecAugment(freq_mask_param=8, time_mask_param=20, n_freq_masks=2, n_time_masks=2)
        x = torch.randn(64, 100)
        out = aug(x)
        assert out.shape == x.shape

    def test_some_values_masked(self) -> None:
        aug = SpecAugment(freq_mask_param=30, time_mask_param=50, n_freq_masks=2, n_time_masks=2)
        torch.manual_seed(0)
        x = torch.ones(64, 100)
        out = aug(x)
        # SpecAugment zeros out some regions
        assert (out == 0).any()

    def test_custom_params(self) -> None:
        aug = SpecAugment(freq_mask_param=4, time_mask_param=10, n_freq_masks=1, n_time_masks=1)
        assert aug.n_freq_masks == 1
        assert aug.n_time_masks == 1
