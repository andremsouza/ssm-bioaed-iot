#!/usr/bin/env bash
# ============================================================================
# quick_experiment.sh — Fast benchmark experiment
#
# Runs a reduced ablation comparing model architectures across datasets,
# then generates statistical analysis and figures.
#
# Modes (mutually exclusive, first flag wins):
#   (default)    quick_pilot    — inceptiontime × aswine × 5 seeds  = 5 runs
#   --extended   extended_pilot — 3 models × 2 datasets × 5 seeds   = 30 runs
#   --full       full_matrix    — 3 models × 2 datasets × 5 seeds   = 30 runs
#
# Options:
#   --skip-ablation   skip training, re-run analysis only
#   --resume          skip runs whose output directory already exists
#
# Usage:
#   bash scripts/quick_experiment.sh
#   bash scripts/quick_experiment.sh --extended
#   bash scripts/quick_experiment.sh --full
#   bash scripts/quick_experiment.sh --skip-ablation
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# Always use the project's own venv python
PYTHON="${PYTHON:-.venv/bin/python}"

# ---------- defaults ----------
ABLATION_PROFILE="quick_pilot"
SKIP_ABLATION=false
RESUME=false

# ---------- parse flags ----------
for arg in "$@"; do
    case "$arg" in
        --full)       ABLATION_PROFILE="full_matrix" ;;
        --extended)   ABLATION_PROFILE="extended_pilot" ;;
        --skip-ablation) SKIP_ABLATION=true ;;
        --resume)     RESUME=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ---------- derive run count from profile ----------
case "$ABLATION_PROFILE" in
    quick_pilot)    TOTAL_RUNS=5   ; LABEL="PILOT    (5 runs, 1 model × 1 dataset × 5 seeds)" ;;
    extended_pilot) TOTAL_RUNS=30  ; LABEL="EXTENDED (30 runs, 3 models × 2 datasets × 5 seeds)" ;;
    full_matrix)    TOTAL_RUNS=30  ; LABEL="FULL     (30 runs, 3 models × 2 datasets × 5 seeds)" ;;
esac

echo "=== Benchmark Experiment ==="
echo "Working directory: $(pwd)"
echo "Mode:    ${LABEL}"
echo "Profile: configs/ablation/${ABLATION_PROFILE}.yaml"
[[ "$RESUME" == true ]]        && echo "Resume:  ON (skipping existing output dirs)"
[[ "$SKIP_ABLATION" == true ]] && echo "Ablation: SKIPPED (analysis only)"
echo ""

# ---------- step 0: preflight validation ----------
if [[ "$SKIP_ABLATION" == false ]]; then
    echo ">>> Step 0/3: Preflight validation (smoke tests for every model × dataset combo)..."
    if ! "$PYTHON" -m bioaed.preflight "${ABLATION_PROFILE}"; then
        echo ""
        echo "!!! Preflight FAILED — fix the errors above before running the experiment."
        exit 1
    fi
    echo ""
fi

# ---------- step 1: ablation runs ----------
if [[ "$SKIP_ABLATION" == false ]]; then
    echo ">>> Step 1/3: Running ablation (${TOTAL_RUNS} runs)..."
    if [[ "$RESUME" == true ]]; then
        "$PYTHON" -m bioaed.ablation +ablation="${ABLATION_PROFILE}" +resume=true --multirun
    else
        "$PYTHON" -m bioaed.ablation +ablation="${ABLATION_PROFILE}" --multirun
    fi
    echo ""
else
    echo ">>> Step 1/3: Ablation skipped."
    echo ""
fi

# ---------- step 2: analysis & figures ----------
echo ">>> Step 3/3: Generating analysis & figures..."
"$PYTHON" -m bioaed.evaluation.report_generator

echo ""
echo "=== Done! ==="
echo ""
echo "Results:"
echo "  outputs/ablation/   — per-run checkpoints & metrics"
echo "  reports/            — statistical reports & figures"
echo ""
if [[ -f reports/stats_mAP.txt ]]; then
    echo "--- Statistical summary (mAP) ---"
    head -30 reports/stats_mAP.txt
fi
