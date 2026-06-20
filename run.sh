#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-kvasir_seg}"
DATA_ROOT="${DATA_ROOT:-data}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SEED="${SEED:-42}"
FAIR_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp,plain_fourier_unet,apdr_fourier_unet"
FAITHFUL_MODELS="unet,unetpp,attention_unet,pranet,acsnet,hardnet_mseg,cfanet,polyp_pvt,caranet,hsnet,resunetpp"

case "${1:-help}" in
  install)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  prepare)
    shift
    "$PYTHON_BIN" scripts/prepare_dataset.py --dataset "$DATASET" --data-root "$DATA_ROOT" "$@"
    ;;
  fair)
    "$PYTHON_BIN" scripts/benchmark_all.py --models "$FAIR_MODELS" --config-dir configs/fair --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$OUTPUT_ROOT"
    ;;
  faithful)
    "$PYTHON_BIN" scripts/benchmark_all.py --models "$FAITHFUL_MODELS" --config-dir configs/official_faithful --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --output-root "$OUTPUT_ROOT"
    ;;
  ablation)
    "$PYTHON_BIN" scripts/run_apdr_ablation.py --dataset "$DATASET" --data-root "$DATA_ROOT" --device "$DEVICE" --seed "$SEED" --output-root "$OUTPUT_ROOT"
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q
    ;;
  smoke)
    "$PYTHON_BIN" scripts/smoke_all_models.py --config-dir configs/fair
    ;;
  *)
    echo "Usage: bash run.sh {install|prepare|fair|faithful|ablation|smoke|test}"
    ;;
esac
