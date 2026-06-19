#!/usr/bin/env bash
#
# Download and arrange the AnuraSet and aSwine datasets into the layout expected
# by the bioaed data loaders (see src/bioaed/data/). After downloading, the
# layout is validated with `python -m bioaed.preflight`.
#
# Datasets are NOT redistributed here. This script points to the official
# sources and expects you to supply the archive URLs via environment variables
# (the exact download links are documented in the README "Data" section):
#
#   AnuraSet  Canas et al. 2023, Scientific Data, doi:10.1038/s41597-023-02666-2
#             Code and download: https://github.com/soundclim/anuraset
#   aSwine    Souza et al. 2025, Applied Intelligence, doi:10.1007/s10489-025-06555-6
#             Dataset: https://github.com/andremsouza/aswine
#
# Usage:
#   ANURASET_URL=<url> ASWINE_URL=<url> bash scripts/download_data.sh [all|anuraset|aswine]
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
mkdir -p "${DATA_DIR}"
echo "==> Target data directory: ${DATA_DIR}"

anuraset() {
  echo "==> AnuraSet -> ${DATA_DIR}/anuraset"
  mkdir -p "${DATA_DIR}/anuraset"
  : "${ANURASET_URL:?Set ANURASET_URL to the official AnuraSet archive (see https://github.com/soundclim/anuraset)}"
  curl -L "${ANURASET_URL}" -o "${DATA_DIR}/anuraset/anuraset_archive"
  # Expected after extraction: ${DATA_DIR}/anuraset/{audio/<site>/*.wav, metadata.csv}
  unzip -n "${DATA_DIR}/anuraset/anuraset_archive" -d "${DATA_DIR}/anuraset" || \
    tar -xf "${DATA_DIR}/anuraset/anuraset_archive" -C "${DATA_DIR}/anuraset"
}

aswine() {
  echo "==> aSwine -> ${DATA_DIR}/aswine"
  mkdir -p "${DATA_DIR}/aswine"
  : "${ASWINE_URL:?Set ASWINE_URL to the aSwine dataset archive (see https://github.com/andremsouza/aswine)}"
  curl -L "${ASWINE_URL}" -o "${DATA_DIR}/aswine/aswine_archive"
  # Expected after extraction: ${DATA_DIR}/aswine/{audio/*.wav, meta/1s_pruned/*.csv}
  unzip -n "${DATA_DIR}/aswine/aswine_archive" -d "${DATA_DIR}/aswine" || \
    tar -xf "${DATA_DIR}/aswine/aswine_archive" -C "${DATA_DIR}/aswine"
}

main() {
  case "${1:-all}" in
    anuraset) anuraset ;;
    aswine) aswine ;;
    all) anuraset; aswine ;;
    *) echo "usage: $0 [all|anuraset|aswine]" >&2; exit 1 ;;
  esac
  echo "==> Validating dataset layout with bioaed.preflight"
  python -m bioaed.preflight
  echo "==> Done."
}

main "$@"
