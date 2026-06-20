# APDR-Fourier U-Net repository

This repository contains eleven medical-image segmentation comparison
baselines and **two proposed spectral segmentation methods**:

1. `plain_fourier_unet` — **Proposal I: Plain Fourier U-Net**;
2. `apdr_fourier_unet` — **Proposal II: APDR-Fourier U-Net**.

All former APF, URF, bounded-Fourier, amplitude-only, phase-only, placement,
no-residual, and no-channel-mixing proposal variants have been removed from the
active registry, configurations, and runners.

## Proposed methods

### Proposal I — Plain Fourier U-Net

Plain Fourier U-Net introduces an identity-initialized Fourier bottleneck that
jointly learns bounded amplitude and phase corrections while preserving a
residual spatial path.

### Proposal II — APDR-Fourier U-Net

**APDR-Fourier U-Net** is an **Amplitude–Phase Disagreement-Routed Residual
Fourier U-Net**:

\[
\text{APDR-Fourier U-Net}
=
\text{Plain Fourier U-Net}
+
\text{zero-initialized local Fourier residual adapter}.
\]

The complete Proposal-I path remains unchanged inside Proposal II. The APDR
adapter is routed by amplitude–phase disagreement, prediction uncertainty,
boundary evidence, and decoder context. Its scalar residual gate is initialized
to zero, so Proposal II begins with exactly the same output as Proposal I.

See `APDR_FOURIER_UNET_GUIDE.md` for the architecture.

## Active models

### Comparison baselines

`unet`, `unetpp`, `attention_unet`, `pranet`, `acsnet`, `hardnet_mseg`,
`cfanet`, `polyp_pvt`, `caranet`, `hsnet`, `resunetpp`.

### Proposed methods

`plain_fourier_unet`, `apdr_fourier_unet`.

## Dataset support

The challenge dataset retained in this revision is:

- `isbi2012` — ISBI 2012 neuronal-structure EM segmentation.


## Installation and checks

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python tools/audit_fairness.py
python scripts/smoke_all_models.py --config-dir configs/fair
```

## Controlled two-proposal ablation

```bash
python scripts/run_apdr_ablation.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --output-root outputs_apdr_ablation
```

For three seeds on Kaggle:

```bash
bash scripts/kaggle_apdr_ablation_etis_3seeds.sh
```

The ablation compares Proposal I and Proposal II under the same data split,
augmentation, optimizer, learning rate, loss, epoch count, threshold, and
random seeds. The only architectural additions in Proposal II use `apdr_*`
configuration keys.

## Full fair benchmark

```bash
bash run.sh fair
```

The exact retained manuscript baseline rows are in
`docs/manuscript_baselines.tex`. A manuscript-ready two-proposal ablation and
ISBI-2012 dataset fragment is in `docs/manuscript_two_proposals.tex`.
