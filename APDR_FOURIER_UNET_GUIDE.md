# Two proposed Fourier U-Net methods

The repository retains two proposed methods for the controlled ablation.

## Proposal I: Plain Fourier U-Net

Plain Fourier U-Net uses an identity-initialized Fourier bottleneck that learns
bounded amplitude and phase transformations while preserving the residual
spatial representation. It is the foundational spectral proposal rather than an
external comparison baseline.

## Proposal II: APDR-Fourier U-Net

**APDR-Fourier U-Net** means **Amplitude–Phase Disagreement-Routed Residual
Fourier U-Net**. It is defined by

\[
\text{APDR-Fourier U-Net}
=
\text{Plain Fourier U-Net}
+
\text{zero-initialized routed residual adapter}.
\]

The complete Plain Fourier path is not replaced or weakened. It retains the
same encoder, Fourier bottleneck, decoder, segmentation head, amplitude range,
phase range, channel mixer, initialization, and training protocol.

At the bottleneck, amplitude-only and phase-only corrections are computed:

\[
R_A=\mathcal F^{-1}(S G_A-S),\qquad
R_P=\mathcal F^{-1}(S e^{i\Phi}-S).
\]

Their spatial disagreement is

\[
D_{AP}=\operatorname{Mean}_c |R_A-R_P|.
\]

The routing map combines disagreement, prediction uncertainty, boundary
evidence, and decoder context:

\[
Q=\sigma\!\left(\operatorname{Router}[D_{AP},4P(1-P),B,F_d]\right).
\]

A windowed middle/high-frequency Fourier adapter predicts a local residual
\(R_L\). The refined decoder feature is

\[
\widetilde F_d=F_d+\beta Q\odot R_L,
\qquad
\beta=\beta_{\max}\tanh(\widehat\beta).
\]

Because \(\widehat\beta=0\) at initialization, \(\beta=0\), and Proposal II
produces exactly the same logits as Proposal I before learning the residual.

## Controlled two-proposal ablation

Only these methods are present in `configs/ablation`:

- `plain_fourier_unet` — Proposal I;
- `apdr_fourier_unet` — Proposal II.

Both use the same data split, augmentation, image size, batch size, optimizer,
learning rate, BCE–Dice loss, epoch count, threshold, and random seeds. The
ablation therefore measures only the additional contribution of the APDR
adapter.

## Dataset scope

ISBI 2012 support is retained through the `isbi2012` dataset key.

## Run

```bash
python -m pytest -q
python scripts/run_apdr_ablation.py \
  --dataset etis \
  --data-root data \
  --device cuda \
  --seed 42
```

Three-seed Kaggle run:

```bash
bash scripts/kaggle_apdr_ablation_etis_3seeds.sh
```
