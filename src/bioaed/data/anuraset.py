"""AnuraSet dataset adapter for neotropical anuran bioacoustics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bioaed.data.audio_dataset import AudioDataset
from bioaed.features.quality_gate import QualityGate


class AnuraSetDataset(AudioDataset):
    """Dataset adapter for the AnuraSet passive acoustic monitoring dataset.

    Reads ``metadata.csv`` (93,378 samples × 42 species) and maps sample names
    to 3-second WAV files in the ``audio/`` directory.

    Args:
        root_dir: Path to the AnuraSet dataset root (containing ``audio/`` and ``metadata.csv``).
        split: Data split — ``"training"`` or ``"test"`` (as defined in the ``subset`` column).
        quality_gate: Optional quality gate for DCAI confidence weighting.
        **kwargs: Additional keyword arguments forwarded to :class:`AudioDataset`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        quality_gate: QualityGate | None = None,
        **kwargs: Any,
    ) -> None:
        # Map our standard split names to AnuraSet's column values
        self._split_map = {"train": "training", "test": "test"}
        super().__init__(
            root_dir=root_dir,
            split=split,
            quality_gate=quality_gate,
            **kwargs,
        )

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load and filter the AnuraSet metadata CSV.

        Returns:
            List of dicts with keys ``fname``, ``site``, ``min_t``, ``max_t``,
            and ``labels`` (list of 42 binary species indicators).
        """
        csv_path = self.root_dir / "metadata.csv"
        if not csv_path.exists():
            msg = f"Metadata CSV not found: {csv_path}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(csv_path)

        # Filter by subset column
        subset_value = self._split_map.get(self.split, self.split)
        df_split = df[df["subset"] == subset_value].reset_index(drop=True)

        # Identify species columns (all columns after 'subset')
        species_start_idx = df_split.columns.get_loc("subset") + 1  # type: ignore[operator]
        species_columns = df_split.columns[species_start_idx:].tolist()
        self._species_columns = species_columns

        rows: list[dict[str, Any]] = []
        for _, record in df_split.iterrows():
            labels = [int(record[col]) for col in species_columns]
            rows.append(
                {
                    "fname": str(record["fname"]),
                    "site": str(record["site"]),
                    "min_t": int(record["min_t"]),
                    "max_t": int(record["max_t"]),
                    "labels": labels,
                }
            )
        return rows

    def _get_audio_path_and_offset(self, idx: int) -> tuple[Path, float]:
        """Return the audio file path and offset for a given sample.

        AnuraSet samples are individual 3-second files stored as
        ``audio/{site}/{fname}_{min_t}_{max_t}.wav``, so offset is always 0.

        Args:
            idx: Index into the metadata list.

        Returns:
            Tuple of (audio_file_path, 0.0).
        """
        row = self.metadata[idx]
        site = row["site"]
        fname = row["fname"]
        min_t = row["min_t"]
        max_t = row["max_t"]
        audio_path = self.root_dir / "audio" / site / f"{fname}_{min_t}_{max_t}.wav"
        return audio_path, 0.0
