"""Abstract base dataset for bioacoustic audio event detection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from bioaed.features.quality_gate import QualityGate
from bioaed.features.transforms import SpecAugment, ZNormalize


class AudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ABC):
    """Abstract base class for bioacoustic audio datasets.

    Subclasses must implement :meth:`_load_metadata` and :meth:`_get_audio_path_and_offset`.

    Each ``__getitem__`` call returns a tuple of:
        - ``spectrogram``: log-mel spectrogram tensor ``(n_mels, time_frames)``.
        - ``labels``: multi-hot binary label tensor ``(num_classes,)``.
        - ``quality_weight``: scalar confidence weight from the DCAI quality gate.
    """

    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int,
        segment_duration: float,
        n_mels: int = 64,
        hop_length: int = 160,
        win_length: int = 400,
        num_classes: int = 7,
        quality_gate: QualityGate | None = None,
        split: str = "train",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_classes = num_classes
        self.quality_gate = quality_gate
        self.split = split
        self.normalize: ZNormalize | None = None
        self.augment: SpecAugment | None = None

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Load metadata rows
        self.metadata: list[dict[str, Any]] = self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load and return metadata rows for the current split.

        Returns:
            List of dicts, each containing at minimum an audio path and labels.
        """

    @abstractmethod
    def _get_audio_path_and_offset(self, idx: int) -> tuple[Path, float]:
        """Return the audio file path and start-offset for a given sample index.

        Args:
            idx: Index into the metadata list.

        Returns:
            Tuple of (audio_file_path, offset_seconds).
        """

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load, transform, and return a single sample.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (spectrogram, labels, quality_weight).
        """
        audio_path, offset = self._get_audio_path_and_offset(idx)
        num_frames = int(self.segment_duration * self.sample_rate)

        # Load raw waveform segment via soundfile directly.
        # torchaudio 2.11+ routes torchaudio.load through torchcodec regardless of
        # the backend kwarg, so we bypass it entirely and use soundfile.
        offset_frames = int(offset * self.sample_rate)
        data, sr = sf.read(
            str(audio_path),
            start=offset_frames,
            frames=num_frames,
            dtype="float32",
            always_2d=True,
        )
        # soundfile returns (frames, channels) → (channels, frames)
        waveform = torch.from_numpy(data.T)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or trim to exact length
        if waveform.shape[-1] < num_frames:
            pad_length = num_frames - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[-1] > num_frames:
            waveform = waveform[:, :num_frames]

        # Compute quality weight
        if self.quality_gate is not None:
            quality_weight = self.quality_gate.compute_confidence_weight(waveform)
        else:
            quality_weight = torch.tensor(1.0)

        # Compute log-mel spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Squeeze channel dimension: (1, n_mels, T) → (n_mels, T)
        log_mel_spec = log_mel_spec.squeeze(0)

        # Apply Z-normalization if configured
        if self.normalize is not None:
            log_mel_spec = self.normalize(log_mel_spec)

        # Apply SpecAugment (training only)
        if self.augment is not None and self.split == "train":
            log_mel_spec = self.augment(log_mel_spec)

        # Extract labels
        row = self.metadata[idx]
        labels = self._extract_labels(row)

        return log_mel_spec, labels, quality_weight

    def _extract_labels(self, row: dict[str, Any]) -> torch.Tensor:
        """Extract multi-hot label tensor from a metadata row.

        Args:
            row: A single metadata row dict.

        Returns:
            Float tensor of shape (num_classes,).
        """
        label_values = row.get("labels", [0] * self.num_classes)
        return torch.tensor(np.array(label_values, dtype=np.float32))
