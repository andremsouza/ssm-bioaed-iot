#!/usr/bin/env bash
#
# End-to-end reproduction of the SBBD 2026 paper results
# ("Efficiency-First Bioacoustic Audio Event Detection Under Resource-Constrained
# Systems"). Runs the full pipeline: HPO -> 5-seed ablation -> profiling -> report.
#
# WARNING: this is GPU-intensive. The full run is 180 HPO trials
# (3 models x 2 datasets x 30) plus 30 ablation runs (x 5 seeds). To reuse the
# committed best hyperparameters and skip HPO, run with SKIP_HPO=1.
#
# Models: ast = AST, audio_mamba = MambaSpec, audio_mamba_pretrained = SSAMBA.
# Profiling numbers in the paper were measured on an NVIDIA RTX 5070 Ti.
#
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="ast,audio_mamba,audio_mamba_pretrained"
DATASETS="aswine,anuraset"
SEEDS="0,1,2,3,4"
N_TRIALS="${N_TRIALS:-30}"

echo "==> [1/6] Install dependencies (CUDA build)"
make install-cuda-full

echo "==> [2/6] Download datasets"
bash scripts/download_data.sh all

echo "==> [3/6] Download pretrained SSAMBA checkpoint"
make download-checkpoints

if [ "${SKIP_HPO:-0}" = "1" ]; then
  echo "==> [4/6] Skipping HPO (SKIP_HPO=1); reusing outputs/hpo/*/best_params.json"
else
  echo "==> [4/6] HPO (Optuna TPE, ${N_TRIALS} trials per model x dataset)"
  uv run python -m bioaed.sweep --multirun \
    model="${MODELS}" dataset="${DATASETS}" +n_trials="${N_TRIALS}"
fi

echo "==> [5/6] 5-seed ablation"
uv run python -m bioaed.ablation --multirun \
  model="${MODELS}" dataset="${DATASETS}" seed="${SEEDS}" +hpo_params_dir=outputs/hpo

echo "==> [5b/6] Efficiency profiling (Table 2)"
uv run python scripts/run_profiling.py

echo "==> [6/6] Generate tables, figures, and statistical tests"
uv run python -m bioaed.evaluation.report_generator

cat <<'EOF'

Reproduction complete. Generated artifacts (in reports/):
  - Table 1 (accuracy) and Table 2 (efficiency):  reports/table_benchmark.tex
  - Pareto and critical-difference figures:        reports/figures/
  - Friedman / Nemenyi statistical tests:          reports/stats_*.txt
EOF
