#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — End-to-end benchmark pipeline
#
# Orchestrates:
#   1. HPO (Optuna) — 30 trials per model×dataset (6 combos)
#   2. Final evaluation — 5 seeds per model×dataset with best HPO params
#   3. Statistical analysis & report generation
#
# Options:
#   --skip-hpo        Skip HPO, use existing best_params.json
#   --skip-ablation   Skip final evaluation, re-run analysis only
#   --resume          Skip runs whose output directory already exists
#   --n-trials N      Number of HPO trials per combo (default: 30)
#   --cleanup-audiomamba    Delete stale scratch audiomamba HPO/ablation + generated reports
#   --hpo-audiomamba-only   Run HPO only for scratch audiomamba on both datasets
#
# Usage:
#   bash scripts/run_pivot_d.sh
#   bash scripts/run_pivot_d.sh --skip-hpo
#   bash scripts/run_pivot_d.sh --resume --n-trials 10
#   bash scripts/run_pivot_d.sh --cleanup-audiomamba
#   bash scripts/run_pivot_d.sh --hpo-audiomamba-only --n-trials 50
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

# ---------- defaults ----------
SKIP_HPO=false
SKIP_ABLATION=false
RESUME=false
N_TRIALS=30
CLEANUP_AUDIOMAMBA=false
HPO_AUDIOMAMBA_ONLY=false
MODELS=(inceptiontime ast audio_mamba audio_mamba_pretrained)
DATASETS=(aswine anuraset)

# Map config names → output directory keys (class names resolved by sweep.py)
declare -A MODEL_KEY_MAP=(
    [inceptiontime]=inceptiontime
    [ast]=audiospectrogramtransformer
    [audio_mamba]=audiomamba
    [audio_mamba_pretrained]=audiomamba_pretrained
)

forbidden_fragment() {
    local path="$1"
    [[ "$path" == *"audiomamba_pretrained"* || "$path" == *"audiospectrogramtransformer"* ]]
}

allowed_delete_scope() {
    local path="$1"
    [[ "$path" == outputs/hpo/audiomamba/* || "$path" == outputs/ablation/audiomamba/* || "$path" == reports/* ]]
}

guard_delete_targets() {
    local target
    for target in "$@"; do
        if forbidden_fragment "$target"; then
            echo "ERROR: Refusing to touch protected path: $target"
            exit 1
        fi
        if ! allowed_delete_scope "$target"; then
            echo "ERROR: Refusing to delete outside allowed scope: $target"
            exit 1
        fi
    done
}

cleanup_audiomamba() {
    echo ">>> Cleanup mode: scratch audiomamba artifacts + generated reports"

    local targets=(
        "outputs/hpo/audiomamba/aswine"
        "outputs/hpo/audiomamba/anuraset"
        "outputs/ablation/audiomamba/aswine/seed_0"
        "outputs/ablation/audiomamba/aswine/seed_1"
        "outputs/ablation/audiomamba/aswine/seed_2"
        "outputs/ablation/audiomamba/aswine/seed_3"
        "outputs/ablation/audiomamba/aswine/seed_4"
        "outputs/ablation/audiomamba/anuraset/seed_0"
        "outputs/ablation/audiomamba/anuraset/seed_1"
        "outputs/ablation/audiomamba/anuraset/seed_2"
        "outputs/ablation/audiomamba/anuraset/seed_3"
        "outputs/ablation/audiomamba/anuraset/seed_4"
    )

    # Generated consolidated report artifacts.
    local glob
    shopt -s nullglob
    for glob in reports/stats_*.txt reports/table_*.tex reports/benchmark_stats_*.txt reports/figures/*.pdf; do
        targets+=("$glob")
    done
    shopt -u nullglob

    guard_delete_targets "${targets[@]}"

    echo "Planned deletions (${#targets[@]} paths):"
    printf '  - %s\n' "${targets[@]}"

    local target
    for target in "${targets[@]}"; do
        if [[ -e "$target" ]]; then
            rm -rf -- "$target"
            echo "  [deleted] $target"
        else
            echo "  [skip] $target (not found)"
        fi
    done
    echo
}

verify_audiomamba_hpo_space() {
    local dataset="$1"
    local db="outputs/hpo/audiomamba/${dataset}/study.db"
    local required=(embed_dim depth d_state patch_size)

    if [[ ! -f "$db" ]]; then
        echo "ERROR: Expected audiomamba study DB not found: $db"
        exit 1
    fi

    local params
    params="$(sqlite3 "$db" "SELECT DISTINCT param_name FROM trial_params ORDER BY param_name;")"
    echo "  [check] audiomamba ${dataset} sampled params:"
    echo "$params" | sed 's/^/         - /'

    local missing=()
    local p
    for p in "${required[@]}"; do
        if ! grep -qx "$p" <<< "$params"; then
            missing+=("$p")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "ERROR: audiomamba study for ${dataset} is missing required architecture params: ${missing[*]}"
        echo "Hint: clean stale audiomamba HPO artifacts first with --cleanup-audiomamba"
        exit 1
    fi
}

# ---------- parse flags ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-hpo)       SKIP_HPO=true; shift ;;
        --skip-ablation)  SKIP_ABLATION=true; shift ;;
        --resume)         RESUME=true; shift ;;
        --n-trials)       N_TRIALS="$2"; shift 2 ;;
        --cleanup-audiomamba) CLEANUP_AUDIOMAMBA=true; shift ;;
        --hpo-audiomamba-only) HPO_AUDIOMAMBA_ONLY=true; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$HPO_AUDIOMAMBA_ONLY" == true ]]; then
    MODELS=(audio_mamba)
    SKIP_ABLATION=true
fi

if [[ "$HPO_AUDIOMAMBA_ONLY" == true && "$SKIP_HPO" == true ]]; then
    echo "ERROR: --hpo-audiomamba-only cannot be combined with --skip-hpo"
    exit 1
fi

if [[ "$CLEANUP_AUDIOMAMBA" == true ]]; then
    cleanup_audiomamba
    echo "Cleanup operation complete. Exiting without running HPO/ablation/report phases."
    exit 0
fi

TOTAL_HPO=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
TOTAL_FINAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * 5 ))

echo "=== Benchmark Pipeline ==="
echo "Working directory: $(pwd)"
echo ""
echo "Phase 1: HPO — ${TOTAL_HPO} combos × ${N_TRIALS} trials each"
echo "Phase 2: Final eval — ${TOTAL_FINAL} runs (${#MODELS[@]} models × ${#DATASETS[@]} datasets × 5 seeds)"
echo "Phase 3: Statistical analysis & reports"
echo ""
[[ "$SKIP_HPO" == true ]]      && echo "HPO:      SKIPPED (using existing best_params.json)"
[[ "$SKIP_ABLATION" == true ]] && echo "Ablation: SKIPPED (analysis only)"
[[ "$RESUME" == true ]]        && echo "Resume:   ON (skipping existing output dirs)"
[[ "$CLEANUP_AUDIOMAMBA" == true ]] && echo "Cleanup:  Completed (scratch audiomamba + generated reports)"
[[ "$HPO_AUDIOMAMBA_ONLY" == true ]] && echo "Mode:     HPO-only for scratch audiomamba"
echo ""

# ================================================================
# Phase 1: HPO sweeps
# ================================================================
if [[ "$SKIP_HPO" == false ]]; then
    echo ">>> Phase 1: Running HPO sweeps (${N_TRIALS} trials each)..."
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            model_key="${MODEL_KEY_MAP[$model]}"
            hpo_out="outputs/hpo/${model_key}/${dataset}"
            if [[ "$RESUME" == true ]] && [[ -f "${hpo_out}/best_params.json" ]]; then
                echo "  [skip] ${model} × ${dataset} — best_params.json exists"
                if [[ "$model" == "audio_mamba" ]]; then
                    verify_audiomamba_hpo_space "$dataset"
                fi
                continue
            fi
            echo "  [run]  ${model} × ${dataset}..."
            "$PYTHON" -m bioaed.sweep \
                model="${model}" \
                dataset="${dataset}" \
                "+n_trials=${N_TRIALS}"
            if [[ "$model" == "audio_mamba" ]]; then
                verify_audiomamba_hpo_space "$dataset"
            fi
        done
    done
    echo ""
fi

# ================================================================
# Phase 2: Final evaluation with HPO params
# ================================================================
if [[ "$SKIP_ABLATION" == false ]]; then
    echo ">>> Phase 2: Running final evaluation (5 seeds per combo)..."
    RESUME_FLAG=""
    # Use ++ (override) because resume=false is already defined in config.yaml
    [[ "$RESUME" == true ]] && RESUME_FLAG="++resume=true"

    "$PYTHON" -m bioaed.ablation \
        +ablation=benchmark \
        +hpo_params_dir=outputs/hpo \
        ${RESUME_FLAG} \
        --multirun
    echo ""
else
    echo ">>> Phase 2: Final evaluation skipped."
    echo ""
fi

# ================================================================
# Phase 3: Analysis & reports
# ================================================================
echo ">>> Phase 3: Generating analysis & figures..."
"$PYTHON" -m bioaed.evaluation.report_generator

echo ""
echo "=== Benchmark Pipeline Complete ==="
echo ""
echo "Results:"
echo "  outputs/hpo/       — HPO trials, best_params.json, study.pkl"
echo "  outputs/ablation/  — per-run checkpoints & metrics"
echo "  reports/           — statistical reports & figures"
echo ""
if [[ -f reports/stats_mAP.txt ]]; then
    echo "--- Statistical summary (mAP) ---"
    head -30 reports/stats_mAP.txt
fi
