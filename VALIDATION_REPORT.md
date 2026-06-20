# Validation report

The revised repository was validated locally with CPU execution.

## Completed checks

- Python compilation: passed for `src`, `scripts`, `tests`, and `tools`.
- Shell syntax: passed for `run.sh` and the Kaggle two-proposal runner.
- Fairness audit: passed with no shared-protocol mismatches.
- Proposal-I/Proposal-II smoke forward and backward: passed.
- Dataset registry: ISBI 2012 retained; the 2018 challenge entry removed.
- Full test suite: **159 passed with no warnings**.

## Ablation scope

The controlled ablation contains exactly two proposed methods:

1. `plain_fourier_unet` — Proposal I;
2. `apdr_fourier_unet` — Proposal II.

Proposal II preserves the complete Proposal-I path and differs only through
`apdr_*` residual-adapter parameters. Both methods use the same split,
augmentation, optimizer, loss, epochs, threshold, and seeds.

## Important result note

No full ETIS or ISBI-2012 training was run during repository editing. Therefore,
this package does not claim that APDR already exceeds the previous Plain Fourier
test score. It is ready for the controlled three-seed experiment in
`scripts/kaggle_apdr_ablation_etis_3seeds.sh`.
