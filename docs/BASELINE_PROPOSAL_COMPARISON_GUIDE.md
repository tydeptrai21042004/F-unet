# Baseline-versus-proposal comparison

The permanent fair comparison contains exactly eleven published baselines and
two proposed methods.

## Baselines

- U-Net
- U-Net++
- Attention U-Net
- PraNet
- ACSNet
- HarDNet-MSEG
- CFANet
- Polyp-PVT
- CaraNet
- HSNet
- ResUNet++

## Proposed methods

- Proposal I: Plain Fourier U-Net
- Proposal II: APDR-Fourier U-Net

All 13 methods use the configurations in `configs/fair`. Their data,
optimization, loss, epoch, threshold, and evaluation sections are identical.
Architecture-specific auxiliary and boundary outputs are disabled in this fair
comparison. The separate `configs/official_faithful` directory retains
paper-style output and auxiliary-loss contracts for implementation auditing.

## Run on ETIS over three seeds

```bash
bash scripts/kaggle_baseline_proposal_etis_3seeds.sh
```

Or run the Python entrypoint directly:

```bash
python scripts/run_baseline_proposal_comparison.py \
  --dataset etis \
  --device cuda \
  --seeds 42,1,2 \
  --allow-insecure-download
```

Generated tables are written to:

```text
outputs_baseline_proposal/etis/results/tables/
├── multi_seed_summary.csv
├── multi_seed_summary.json
├── multi_seed_summary.tex
├── baseline_proposal_training_summary.csv
├── baseline_proposal_comparison.tex
└── baseline_proposal_deltas.csv
```

## Audits

```bash
python tools/audit_baseline_proposal_comparison.py
python tools/audit_baseline_implementations.py
python tools/audit_repository_cleanliness.py
```
