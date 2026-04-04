"""aSwine dataset adapter for weakly-labeled swine barn audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bioaed.data.audio_dataset import AudioDataset
from bioaed.features.quality_gate import QualityGate

# Ordered label columns in the aSwine metadata CSVs
ASWINE_LABEL_COLUMNS: list[str] = [
    "Limpeza de Baia",
    "Fala Humana",
    "Alimentação de Baias",
    "OutrosH",
    "EstresseDisputas",
    "TosseEspirro",
    "OutrosA",
]


class ASwineDataset(AudioDataset):
    """Dataset adapter for the aSwine weakly-labeled swine barn audio.

    Reads metadata from ``meta/{meta_variant}/`` CSVs with 1-second segments
    sliced from 6-minute WAV recordings at 16 kHz mono.

    Args:
        root_dir: Path to the aSwine dataset root (containing ``audio/`` and ``meta/``).
        meta_variant: Which metadata variant to use (``raw``, ``1s``, ``1s_pruned``).
        split: Data split — ``"train"`` or ``"test"``.
        quality_gate: Optional quality gate for DCAI confidence weighting.
        **kwargs: Additional keyword arguments forwarded to :class:`AudioDataset`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        meta_variant: str = "1s_pruned",
        split: str = "train",
        quality_gate: QualityGate | None = None,
        **kwargs: Any,
    ) -> None:
        self.meta_variant = meta_variant
        super().__init__(
            root_dir=root_dir,
            split=split,
            quality_gate=quality_gate,
            **kwargs,
        )

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load the aSwine metadata CSV for the current split.

        Returns:
            List of dicts with keys ``audio_file_path``, ``offset``, ``duration``,
            and ``labels`` (list of 7 binary values).
        """
        meta_dir = self.root_dir / "meta" / self.meta_variant
        csv_path = meta_dir / f"aswine_{self.meta_variant}_{self.split}.csv"

        if not csv_path.exists():
            msg = f"Metadata CSV not found: {csv_path}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(csv_path)
        rows: list[dict[str, Any]] = []
        for _, record in df.iterrows():
            labels = [int(record[col]) for col in ASWINE_LABEL_COLUMNS]
            rows.append(
                {
                    "audio_file_path": str(record["audio_file_path"]),
                    "offset": float(record["offset"]),
                    "duration": float(record["duration"]),
                    "labels": labels,
                }
            )
        return rows

    def _get_audio_path_and_offset(self, idx: int) -> tuple[Path, float]:
        """Return the audio file path and offset for a given sample.

        Args:
            idx: Index into the metadata list.

        Returns:
            Tuple of (absolute_audio_path, offset_seconds).
        """
        row = self.metadata[idx]
        audio_path = self.root_dir / "audio" / Path(row["audio_file_path"]).name
        offset = row["offset"]
        return audio_path, offset
