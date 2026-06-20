#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/kaggle/working"
REPO_DIR="${WORK_ROOT}/F-unet"
REPO_URL="https://github.com/tydeptrai21042004/F-unet.git"

cd "${WORK_ROOT}"
rm -rf "${REPO_DIR}"
git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

bash scripts/kaggle_full_component_ablation_etis_3seeds.sh
