#!/usr/bin/env bash
#
# Download and arrange the AnuraSet and aSwine datasets into the layout expected
# by the bioaed data loaders (see src/bioaed/data/). After downloading, the
# layout is validated with `python -m bioaed.preflight`.
#
# Datasets are NOT redistributed here; the official sources are used by default
# (override either URL via the environment variables shown below):
#
#   AnuraSet  Canas et al. 2023, Scientific Data, doi:10.1038/s41597-023-02666-2
#             Source: https://github.com/soundclim/anuraset (Zenodo record 8342596)
#   aSwine    Souza et al. 2025, Applied Intelligence, doi:10.1007/s10489-025-06555-6
#             Source: https://github.com/andremsouza/aswine
#
# Usage:
#   bash scripts/download_data.sh [all|anuraset|aswine]
#   ANURASET_URL=<url> ASWINE_URL=<url> bash scripts/download_data.sh all   # override
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
ANURASET_URL="${ANURASET_URL:-https://zenodo.org/record/8342596/files/anuraset.zip?download=1}"
ASWINE_URL="${ASWINE_URL:-https://github.com/andremsouza/aswine/archive/refs/heads/main.zip}"
mkdir -p "${DATA_DIR}"
echo "==> Target data directory: ${DATA_DIR}"

anuraset() {
  echo "==> AnuraSet -> ${DATA_DIR}/anuraset"
  mkdir -p "${DATA_DIR}/anuraset"
  curl -fL --retry 3 --retry-all-errors "${ANURASET_URL}" -o "${DATA_DIR}/anuraset/anuraset_archive"
  unzip -n "${DATA_DIR}/anuraset/anuraset_archive" -d "${DATA_DIR}/anuraset" || \
    tar -xf "${DATA_DIR}/anuraset/anuraset_archive" -C "${DATA_DIR}/anuraset"
  # The loader expects ${DATA_DIR}/anuraset/{audio/<site>/*.wav, metadata.csv}.
  # If the Zenodo archive uses a different layout, follow the preprocessing in
  # https://github.com/soundclim/anuraset to produce the 3-second segments.
}

aswine() {
  echo "==> aSwine -> ${DATA_DIR}/aswine"
  mkdir -p "${DATA_DIR}/aswine"
  curl -fL --retry 3 --retry-all-errors "${ASWINE_URL}" -o "${DATA_DIR}/aswine/aswine_archive"
  unzip -n "${DATA_DIR}/aswine/aswine_archive" -d "${DATA_DIR}/aswine" || \
    tar -xf "${DATA_DIR}/aswine/aswine_archive" -C "${DATA_DIR}/aswine"
  # The GitHub archive unpacks into an aswine-main/ subfolder; flatten it so the
  # loader finds ${DATA_DIR}/aswine/{audio/*.wav, meta/1s_pruned/*.csv}.
  if [ -d "${DATA_DIR}/aswine/aswine-main" ]; then
    cp -RT "${DATA_DIR}/aswine/aswine-main" "${DATA_DIR}/aswine"
    rm -rf "${DATA_DIR}/aswine/aswine-main"
  fi
}

main() {
  case "${1:-all}" in
    anuraset) anuraset ;;
    aswine) aswine ;;
    all) anuraset; aswine ;;
    *) echo "usage: $0 [all|anuraset|aswine]" >&2; exit 1 ;;
  esac
  echo "==> Validating dataset layout with bioaed.preflight (profile: ${PREFLIGHT_PROFILE:-extended_pilot})"
  python -m bioaed.preflight "${PREFLIGHT_PROFILE:-extended_pilot}"
  echo "==> Done."
}

main "$@"
