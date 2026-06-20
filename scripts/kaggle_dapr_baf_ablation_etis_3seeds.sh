#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

python scripts/run_dapr_baf_ablation.py \
  --dataset etis \
  --data-root data \
  --image-size 352 \
  --batch-size 6 \
  --epochs 30 \
  --lr 0.0003 \
  --device cuda \
  --num-workers 2 \
  --seeds 42,1,2 \
  --output-root outputs_dapr_baf_ablation/etis
