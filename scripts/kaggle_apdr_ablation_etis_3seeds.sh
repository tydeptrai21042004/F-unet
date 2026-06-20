#!/usr/bin/env bash
set -euo pipefail

# Controlled two-proposal ablation: same ETIS split, training protocol and seeds.
# Proposal I: Plain Fourier U-Net. Proposal II: APDR-Fourier U-Net.

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

MODELS="plain_fourier_unet,apdr_fourier_unet"
SEEDS="42,1,2"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_apdr_ablation/etis}"

python -m pytest -q \
  tests/test_apdr_fourier_unet.py \
  tests/test_repository_scope_and_fairness.py \
  tests/test_pipeline_contracts.py

python scripts/benchmark_multi_seed.py \
  --models "$MODELS" \
  --config-dir configs/ablation \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds "$SEEDS" \
  --output-root "$OUTPUT_ROOT" \
  --allow-insecure-download \
  --delete-checkpoints-after-eval

python - "$OUTPUT_ROOT/results/tables/multi_seed_summary.csv" <<'PY'
import csv, sys
from pathlib import Path
path=Path(sys.argv[1])
rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
expected={'plain_fourier_unet','apdr_fourier_unet'}
found={r['model'] for r in rows}
if found != expected:
    raise SystemExit(f'Expected {expected}, found {found}')
for row in rows:
    print(row['model'], 'Dice:', row['dice_mean_pm_std'], 'IoU:', row['iou_mean_pm_std'])
PY
