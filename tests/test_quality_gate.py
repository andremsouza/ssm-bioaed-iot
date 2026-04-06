"""Tests for the DCAI quality gate module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bioaed.features.quality_gate import QualityGate


class TestQualityGate:
    """Tests for the QualityGate class."""

    def test_compute_snr_pure_tone(self) -> None:
        """A pure sine wave should have a measurable SNR."""
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        waveform = torch.from_numpy(sine).unsqueeze(0)

        snr = QualityGate.compute_snr(waveform)
        # Pure tone should have positive SNR
        assert snr > 0.0

    def test_compute_snr_noise(self) -> None:
        """White noise should have low SNR (close to 0)."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16000).astype(np.float32)
        waveform = torch.from_numpy(noise).unsqueeze(0)

        snr = QualityGate.compute_snr(waveform)
        # White noise SNR should be close to 0
        assert abs(snr) < 10.0

    def test_compute_spectral_flatness_noise(self) -> None:
        """White noise should have high spectral flatness (~1.0)."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16000).astype(np.float32)
        waveform = torch.from_numpy(noise).unsqueeze(0)

        flatness = QualityGate.compute_spectral_flatness(waveform)
        assert 0.0 < flatness <= 1.0
        # White noise should be relatively flat
        assert flatness > 0.3

    def test_compute_spectral_flatness_tone(self) -> None:
        """A pure sine wave should have low spectral flatness."""
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        waveform = torch.from_numpy(sine).unsqueeze(0)

        flatness = QualityGate.compute_spectral_flatness(waveform)
        assert 0.0 <= flatness < 0.3

    def test_confidence_weight_soft(self) -> None:
        """Soft weighting should return a float in (0, 1)."""
        gate = QualityGate(
            snr_threshold=5.0,
            spectral_flatness_threshold=0.5,
            weighting_strategy="soft",
        )

        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        waveform = torch.from_numpy(sine).unsqueeze(0)

        weight = gate.compute_confidence_weight(waveform)
        assert isinstance(weight, torch.Tensor)
        assert 0.0 <= weight.item() <= 1.0

    def test_confidence_weight_hard(self) -> None:
        """Hard weighting should return exactly 0.0 or 1.0."""
        gate = QualityGate(
            snr_threshold=5.0,
            spectral_flatness_threshold=0.5,
            weighting_strategy="hard",
        )

        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16000).astype(np.float32)
        waveform = torch.from_numpy(noise).unsqueeze(0)

        weight = gate.compute_confidence_weight(waveform)
        assert weight.item() in (0.0, 1.0)


class TestQualityGateEdgeCases:
    """Edge-case coverage for quality gate branches."""

    def test_compute_snr_very_short_audio_returns_zero(self) -> None:
        """Audio shorter than 5 energy frames must return SNR=0.0."""
        # 100 samples → frame_length=100, hop=25 → 1 frame < 5 → early return
        waveform = torch.randn(1, 100)
        snr = QualityGate.compute_snr(waveform)
        assert snr == pytest.approx(0.0)

    def test_flatness_limit_zero_threshold_is_one(self) -> None:
        """When spectral_flatness_threshold=0, _flatness_limit should be 1.0."""
        gate = QualityGate(
            snr_threshold=5.0,
            spectral_flatness_threshold=0.0,
            weighting_strategy="hard",
        )
        assert gate._flatness_limit == pytest.approx(1.0)

    def test_flatness_limit_positive_threshold_is_threshold(self) -> None:
        """When spectral_flatness_threshold>0, _flatness_limit should equal the threshold."""
        gate = QualityGate(
            snr_threshold=5.0,
            spectral_flatness_threshold=0.3,
            weighting_strategy="hard",
        )
        assert gate._flatness_limit == pytest.approx(0.3)

