"""Reusable audio transforms: resampling, log-mel spectrogram, normalization."""

from __future__ import annotations

import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    """Compute a log-scaled mel spectrogram from a raw waveform.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_mels: Number of mel filter banks.
        n_fft: FFT window size.
        hop_length: Hop length between STFT frames.
        win_length: Window length for STFT.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Args:
            waveform: Raw audio tensor of shape ``(batch, 1, samples)``
                or ``(1, samples)``.

        Returns:
            Log-mel spectrogram of shape ``(..., n_mels, time_frames)``.
        """
        mel = self.mel_spectrogram(waveform)
        return torch.log(mel + 1e-9)


class ZNormalize(torch.nn.Module):
    """Z-normalize a spectrogram using pre-computed per-channel statistics.

    Args:
        mean: Per-channel mean tensor of shape ``(n_mels, 1)``.
        std: Per-channel std tensor of shape ``(n_mels, 1)``.
    """

    def __init__(
        self,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean if mean is not None else torch.tensor(0.0))
        self.register_buffer("std", std if std is not None else torch.tensor(1.0))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply Z-normalization.

        Args:
            spectrogram: Input tensor of shape ``(..., n_mels, time_frames)``.

        Returns:
            Normalized tensor of the same shape.
        """
        return (spectrogram - self.mean) / (self.std + 1e-9)  # type: ignore[operator]

    @classmethod
    def from_dataset_stats(cls, mean: torch.Tensor, std: torch.Tensor) -> ZNormalize:
        """Create a ZNormalize instance from dataset-level statistics.

        Args:
            mean: Mean tensor of shape ``(n_mels, 1)``.
            std: Standard deviation tensor of shape ``(n_mels, 1)``.

        Returns:
            Configured ZNormalize transform.
        """
        return cls(mean=mean, std=std)


class SpecAugment(torch.nn.Module):
    """SpecAugment data augmentation for spectrograms.

    Applies frequency and time masking as described in
    *SpecAugment: A Simple Data Augmentation Method for ASR*.

    Args:
        freq_mask_param: Maximum width of frequency masks.
        time_mask_param: Maximum width of time masks.
        n_freq_masks: Number of frequency masks to apply.
        n_time_masks: Number of time masks to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a spectrogram.

        Args:
            spectrogram: Input tensor of shape ``(..., n_mels, time_frames)``.

        Returns:
            Augmented spectrogram of the same shape.
        """
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_masking(spectrogram)
        for _ in range(self.n_time_masks):
            spectrogram = self.time_masking(spectrogram)
        return spectrogram
