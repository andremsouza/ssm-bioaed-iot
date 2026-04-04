"""Data-Centric AI quality gate for bioacoustic samples.

Computes non-destructive acoustic quality metrics (SNR, spectral flatness)
and maps them to confidence weights for the training loss function.
"""

from __future__ import annotations

from typing import Literal

import librosa
import numpy as np
import torch


class QualityGate:
    """Non-destructive quality gate for bioacoustic audio segments.

    Instead of filtering or denoising the audio (which causes Enhancement-Induced
    Degradation), this gate computes objective quality metrics and returns a
    confidence weight in ``[0, 1]`` for use in the training loss.

    Args:
        snr_threshold: Minimum SNR (dB) below which samples receive reduced weight.
        spectral_flatness_threshold: Maximum spectral flatness above which samples
            receive reduced weight (high flatness = noise-like).
        weighting_strategy: ``"soft"`` for continuous sigmoid-based weights,
            ``"hard"`` for binary inclusion/exclusion.
    """

    def __init__(
        self,
        snr_threshold: float = 0.0,
        spectral_flatness_threshold: float = 0.0,
        weighting_strategy: Literal["soft", "hard"] = "soft",
        alpha: float = 0.5,
        beta: float = 10.0,
    ) -> None:
        self.snr_threshold = snr_threshold
        self.spectral_flatness_threshold = spectral_flatness_threshold
        self.weighting_strategy = weighting_strategy
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def compute_snr(waveform: torch.Tensor) -> float:
        """Estimate the Signal-to-Noise Ratio of a waveform segment.

        Uses a simple energy-based heuristic: signal energy in the top 20% frames
        vs. noise energy in the bottom 20%.

        Args:
            waveform: Audio tensor of shape ``(1, num_samples)`` or ``(num_samples,)``.

        Returns:
            Estimated SNR in decibels.
        """
        audio = waveform.squeeze().numpy()
        # Frame-wise energy
        frame_length = min(2048, len(audio))
        hop = frame_length // 4
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop)
        energies = np.sum(frames**2, axis=0)

        if len(energies) < 5:
            return 0.0

        sorted_energies = np.sort(energies)
        n = len(sorted_energies)
        noise_energy = np.mean(sorted_energies[: max(1, n // 5)]) + 1e-10
        signal_energy = np.mean(sorted_energies[-max(1, n // 5) :]) + 1e-10

        snr_db: float = 10.0 * np.log10(signal_energy / noise_energy)
        return snr_db

    @staticmethod
    def compute_spectral_flatness(waveform: torch.Tensor) -> float:
        """Compute the mean spectral flatness (Wiener entropy) of a waveform.

        Values close to 1.0 indicate noise-like signals, values close to 0.0
        indicate tonal signals.

        Args:
            waveform: Audio tensor of shape ``(1, num_samples)`` or ``(num_samples,)``.

        Returns:
            Mean spectral flatness in ``[0, 1]``.
        """
        audio = waveform.squeeze().numpy().astype(np.float32)
        flatness = librosa.feature.spectral_flatness(y=audio)
        return float(np.mean(flatness))

    def compute_confidence_weight(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map acoustic quality metrics to a confidence weight for the loss.

        Args:
            waveform: Audio tensor of shape ``(1, num_samples)`` or ``(num_samples,)``.

        Returns:
            Scalar tensor with confidence weight in ``[0, 1]``.
        """
        snr = self.compute_snr(waveform)
        flatness = self.compute_spectral_flatness(waveform)

        if self.weighting_strategy == "hard":
            # Binary: include (1.0) or exclude (0.0)
            weight = (
                1.0 if (snr >= self.snr_threshold and flatness <= self._flatness_limit) else 0.0
            )
        else:
            # Soft: sigmoid-based continuous weighting
            # Higher SNR → higher weight, higher flatness → lower weight
            snr_score = _sigmoid(snr - self.snr_threshold, scale=self.alpha)
            flatness_score = _sigmoid(self.spectral_flatness_threshold - flatness, scale=self.beta)
            weight = snr_score * flatness_score

        return torch.tensor(weight, dtype=torch.float32)

    @property
    def _flatness_limit(self) -> float:
        """Effective flatness limit for hard gating."""
        return self.spectral_flatness_threshold if self.spectral_flatness_threshold > 0 else 1.0


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Numerically stable sigmoid function.

    Args:
        x: Input value.
        scale: Scaling factor applied to x before sigmoid.

    Returns:
        Sigmoid output in (0, 1).
    """
    z = scale * x
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    exp_z = np.exp(z)
    return float(exp_z / (1.0 + exp_z))
