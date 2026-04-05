#!/usr/bin/env python
"""
Empirically measure SNR and spectral flatness distributions across a dataset
and recommend DCQG threshold values.

Usage:
    python scripts/calibrate_quality_gate.py --dataset aswine
    python scripts/calibrate_quality_gate.py --dataset anuraset --samples 2000

Reads the same dataset root paths defined in configs/dataset/*.yaml so it
can be run without a full Hydra launch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import soundfile as sf
import torch
from loguru import logger

from bioaed.features.quality_gate import QualityGate


# ── dataset root resolution ──────────────────────────────────────────────────


def _load_dataset_cfg(dataset_name: str) -> dict:
    import os
    import re

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "dataset" / f"{dataset_name}.yaml"
    with open(cfg_path) as f:
        text = f.read()

    # Resolve ${oc.env:VAR,default} interpolations
    def _resolve(m):
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default)

    text = re.sub(r"\$\{oc\.env:(\w+),([^}]+)\}", _resolve, text)
    return yaml.safe_load(text)


def _iter_aswine_wavs(root: Path, max_samples: int):
    """Yield (path, offset_s, duration_s) tuples from aSwine 1s_pruned train split."""
    import pandas as pd

    meta_dir = root / "meta" / "1s_pruned"
    csv_files = sorted(meta_dir.glob("*train*.csv")) or sorted(meta_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found under {meta_dir}")
    rows = []
    for f in csv_files:
        rows.append(pd.read_csv(f))
    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1, random_state=0).head(max_samples)
    for _, row in df.iterrows():
        # audio_file_path in the CSV is e.g. "./data/audio/foo.wav" — just the basename lives under root/audio/
        basename = Path(row["audio_file_path"]).name
        audio_path = root / "audio" / basename
        if audio_path.exists():
            yield str(audio_path), float(row["offset"]), float(row["duration"])


def _iter_anuraset_wavs(root: Path, max_samples: int):
    """Yield (path, offset_s, duration_s) tuples from AnuraSet metadata."""
    import pandas as pd

    csv_path = root / "metadata.csv"
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=0).head(max_samples)
    for _, row in df.iterrows():
        # Files are stored as audio/{site}/{fname}_{min_t}_{max_t}.wav, offset always 0
        audio_path = (
            root / "audio" / row["site"] / f"{row['fname']}_{row['min_t']}_{row['max_t']}.wav"
        )
        if audio_path.exists():
            yield str(audio_path), 0.0, float(row["max_t"] - row["min_t"])


# ── main analysis ─────────────────────────────────────────────────────────────


def analyse(dataset_name: str, max_samples: int = 1000) -> None:
    cfg = _load_dataset_cfg(dataset_name)
    root = Path(cfg.get("root_dir", f"data/{dataset_name}"))
    sample_rate = int(cfg.get("sample_rate", 16000))
    segment_duration = float(cfg.get("segment_duration", 1.0))
    num_frames = int(segment_duration * sample_rate)

    logger.info(
        f"Dataset: {dataset_name}  root={root}  sr={sample_rate}  "
        f"segment={segment_duration}s  samples={max_samples}"
    )

    if dataset_name == "aswine":
        iterator = _iter_aswine_wavs(root, max_samples)
    elif dataset_name == "anuraset":
        iterator = _iter_anuraset_wavs(root, max_samples)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    snrs, flatnesses, weights_default = [], [], []
    gate_default = QualityGate(
        snr_threshold=0.0, spectral_flatness_threshold=0.0, alpha=0.5, beta=10.0
    )

    n = 0
    for audio_path, offset, duration in iterator:
        try:
            offset_frames = int(offset * sample_rate)
            data, sr = sf.read(
                audio_path, start=offset_frames, frames=num_frames, dtype="float32", always_2d=True
            )
            waveform = torch.from_numpy(data.T)
            if waveform.shape[-1] < num_frames:
                waveform = torch.nn.functional.pad(waveform, (0, num_frames - waveform.shape[-1]))
            else:
                waveform = waveform[:, :num_frames]

            snr = QualityGate.compute_snr(waveform)
            flat = QualityGate.compute_spectral_flatness(waveform)
            w = gate_default.compute_confidence_weight(waveform).item()

            snrs.append(snr)
            flatnesses.append(flat)
            weights_default.append(w)
            n += 1
        except Exception as e:
            logger.warning(f"Skipping {audio_path}: {e}")

    if n == 0:
        logger.error("No samples processed — check dataset path.")
        return

    snrs_arr = np.array(snrs)
    flat_arr = np.array(flatnesses)
    w_arr = np.array(weights_default)

    def pct(arr, p):
        return float(np.percentile(arr, p))

    logger.info("\n── SNR distribution (dB) ──────────────────────────────")
    logger.info(f"  n={n}  mean={snrs_arr.mean():.2f}  std={snrs_arr.std():.2f}")
    logger.info(
        f"  p5={pct(snrs_arr, 5):.2f}  p25={pct(snrs_arr, 25):.2f}  "
        f"p50={pct(snrs_arr, 50):.2f}  p75={pct(snrs_arr, 75):.2f}  "
        f"p95={pct(snrs_arr, 95):.2f}"
    )

    logger.info("\n── Spectral flatness distribution ─────────────────────")
    logger.info(f"  mean={flat_arr.mean():.4f}  std={flat_arr.std():.4f}")
    logger.info(
        f"  p5={pct(flat_arr, 5):.4f}  p25={pct(flat_arr, 25):.4f}  "
        f"p50={pct(flat_arr, 50):.4f}  p75={pct(flat_arr, 75):.4f}  "
        f"p95={pct(flat_arr, 95):.4f}"
    )

    logger.info("\n── Quality weights with DEFAULT thresholds (snr=0, flat=0) ─")
    logger.info(
        f"  mean={w_arr.mean():.4f}  std={w_arr.std():.4f}  "
        f"p10={pct(w_arr, 10):.4f}  p50={pct(w_arr, 50):.4f}  "
        f"p90={pct(w_arr, 90):.4f}"
    )
    low_pct = (w_arr < 0.3).mean() * 100
    logger.info(
        f"  samples with weight < 0.3: {low_pct:.1f}%  "
        f"(ideally only noisy samples should be this low)"
    )

    # Recommend thresholds: SNR threshold = p25 (clean = top 75%),
    # flatness threshold = p75 (noise-like = top 25%)
    snr_thresh_rec = round(float(pct(snrs_arr, 25)), 1)
    flat_thresh_rec = round(float(pct(flat_arr, 75)), 4)

    # Simulate weights with recommended thresholds
    gate_rec = QualityGate(
        snr_threshold=snr_thresh_rec,
        spectral_flatness_threshold=flat_thresh_rec,
        alpha=0.5,
        beta=10.0,
    )
    w_rec = np.array(
        [
            gate_rec.compute_confidence_weight(
                torch.nn.functional.pad(
                    torch.from_numpy(
                        sf.read(
                            p,
                            start=int(o * sample_rate),
                            frames=num_frames,
                            dtype="float32",
                            always_2d=True,
                        )[0].T
                    ),
                    (0, max(0, num_frames - int(sf.info(p).frames - int(o * sample_rate)))),
                )[:, :num_frames]
            ).item()
            for p, o, _ in (
                list(
                    _iter_aswine_wavs(root, max_samples)
                    if dataset_name == "aswine"
                    else _iter_anuraset_wavs(root, max_samples)
                )[:n]
            )
            if Path(p).exists()
        ]
        or [0.0]
    )

    logger.info("\n── Recommended thresholds ─────────────────────────────")
    logger.info(f"  snr_threshold:               {snr_thresh_rec}  (p25 of dataset SNR)")
    logger.info(f"  spectral_flatness_threshold: {flat_thresh_rec}  (p75 of dataset flatness)")
    logger.info(
        f"  → With these, weight mean={np.mean(w_rec):.4f}  std={np.std(w_rec):.4f}  "
        f"p10={np.percentile(w_rec, 10):.4f}  p90={np.percentile(w_rec, 90):.4f}"
    )

    logger.info(f"\n  Paste into configs/quality_gate/default.yaml:")
    logger.info(f"    snr_threshold: {snr_thresh_rec}")
    logger.info(f"    spectral_flatness_threshold: {flat_thresh_rec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate DCQG thresholds for a dataset.")
    parser.add_argument("--dataset", choices=["aswine", "anuraset"], default="aswine")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to measure (default: 1000)"
    )
    args = parser.parse_args()
    analyse(args.dataset, args.samples)
