#!/usr/bin/env bash
# ============================================================================
# quick_experiment.sh — Fast DCQG hypothesis test
#
# Runs a reduced ablation (20 epochs, 3 seeds) comparing DCQG on vs off,
# then generates statistical analysis and figures.
#
# Usage:
#   bash scripts/quick_experiment.sh                # defaults: inceptiontime + aswine
#   bash scripts/quick_experiment.sh --full         # all 3 models × 2 datasets (60 runs)
#   EPOCHS=50 SEEDS=5 bash scripts/quick_experiment.sh  # override via env vars
# ============================================================================
set -euo pipefail

# ---------- configurable via environment variables ----------
MODEL="${MODEL:-inceptiontime}"
DATASET="${DATASET:-aswine}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEEDS="${SEEDS:-0,1,2}"
PRECISION="${PRECISION:-32}"
FULL="${1:-}"

cd "$(dirname "$0")/.."
echo "=== Quick DCQG Experiment ==="
echo "Working directory: $(pwd)"
echo ""

if [[ "$FULL" == "--full" ]]; then
    MODEL="inceptiontime,ast,audio_mamba"
    DATASET="aswine,anuraset"
    SEEDS="0,1,2,3,4"
    n_models=3; n_datasets=2; n_seeds=5
    total=$((n_models * n_datasets * 2 * n_seeds))
    echo "Mode:    FULL (${total} runs)"
else
    IFS=',' read -ra _m <<< "$MODEL"; n_models=${#_m[@]}
    IFS=',' read -ra _d <<< "$DATASET"; n_datasets=${#_d[@]}
    IFS=',' read -ra _s <<< "$SEEDS"; n_seeds=${#_s[@]}
    total=$((n_models * n_datasets * 2 * n_seeds))
    echo "Mode:    PILOT (${total} runs)"
fi

echo "Models:  ${MODEL}"
echo "Datasets:${DATASET}"
echo "Seeds:   ${SEEDS}"
echo "Epochs:  ${EPOCHS} (patience=${PATIENCE})"
echo "Prec:    ${PRECISION}"
echo ""

# ---------- step 1: ablation runs ----------
echo ">>> Step 1/3: Running ablation..."
python -m bioaed.ablation --multirun \
    model="${MODEL}" \
    dataset="${DATASET}" \
    quality_gate.enabled=true,false \
    training.max_epochs="${EPOCHS}" \
    training.patience="${PATIENCE}" \
    training.warmup_epochs=2 \
    training.precision="${PRECISION}" \
    seed="${SEEDS}"

echo ""
echo ">>> Step 2/3: Generating analysis & figures..."
python -m bioaed.evaluation.report_generator

echo ""
echo ">>> Step 3/3: Done!"
echo ""
echo "Results:"
echo "  outputs/ablation/   — per-run checkpoints & metrics"
echo "  reports/             — statistical reports & figures"
echo ""
if [[ -f reports/stats_mAP.txt ]]; then
    echo "--- Statistical summary (mAP) ---"
    head -30 reports/stats_mAP.txt
fi
