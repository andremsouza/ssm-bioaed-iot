#!/usr/bin/env bash
# ============================================================================
# run_mamba_pretrained.sh — SSAMBA pretrained benchmark
#
# Runs the SSAMBA-tiny (24-layer BiMamba v2) pretrained model on both
# datasets for 5 seeds each (10 runs total).
#
# Options:
#   --resume          skip runs whose output directory already exists
#   --skip-training   skip training, re-run analysis only
#
# Usage:
#   bash scripts/run_mamba_pretrained.sh
#   bash scripts/run_mamba_pretrained.sh --resume
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SKIP_TRAINING=false
RESUME=false

for arg in "$@"; do
    case "$arg" in
        --skip-training) SKIP_TRAINING=true ;;
        --resume)        RESUME=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo "=== SSAMBA Pretrained Benchmark ==="
echo "Working directory: $(pwd)"
echo "Checkpoint: checkpoints/ssamba_tiny_400.pth"
echo "Runs: 2 datasets × 5 seeds = 10"
[[ "$RESUME" == true ]]         && echo "Resume:   ON"
[[ "$SKIP_TRAINING" == true ]]  && echo "Training: SKIPPED"
echo ""

# ---------- verify checkpoint ----------
if [[ ! -f checkpoints/ssamba_tiny_400.pth ]]; then
    echo "ERROR: SSAMBA checkpoint not found at checkpoints/ssamba_tiny_400.pth"
    echo "Download it first (see README or run the HuggingFace download snippet)."
    exit 1
fi

# ---------- training ----------
if [[ "$SKIP_TRAINING" == false ]]; then
    echo ">>> Running 10 ablation runs..."
    CMD="$PYTHON -m bioaed.ablation +ablation=mamba_pretrained_benchmark"
    if [[ "$RESUME" == true ]]; then
        CMD="$CMD resume=true"
    fi
    eval "$CMD"
    echo ""
else
    echo ">>> Training skipped."
    echo ""
fi

echo "=== Done! ==="
echo ""
echo "Results saved to: outputs/pretrained_benchmark/"
