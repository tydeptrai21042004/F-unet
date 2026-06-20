# Clean source and baseline/proposal validation report

## Repository cleanup

- Removed every `__pycache__` directory and `.pyc` file from the source tree.
- Removed `.pytest_cache` and other generated cache directories.
- Confirmed that no checkpoints, arrays, archives, images, PDFs, shared
  libraries, or other binary artifacts remain inside the repository.
- Added `.gitignore` rules for bytecode, caches, outputs, datasets,
  checkpoints, model weights, archives, and generated media.
- Added `scripts/clean_repository_artifacts.py` for previewing or deleting
  generated artifacts.
- Added `tools/audit_repository_cleanliness.py` to fail when cache or binary
  artifacts are present.

## Baseline-versus-proposal support

The permanent comparison contains exactly:

- 11 published baselines: U-Net, U-Net++, Attention U-Net, PraNet, ACSNet,
  HarDNet-MSEG, CFANet, Polyp-PVT, CaraNet, HSNet, and ResUNet++;
- Proposal I: Plain Fourier U-Net;
- Proposal II: APDR-Fourier U-Net.

Added permanent support files:

- `scripts/baseline_proposal_spec.py`
- `scripts/run_baseline_proposal_comparison.py`
- `scripts/report_baseline_proposal_comparison.py`
- `scripts/kaggle_baseline_proposal_etis_3seeds.sh`
- `tools/audit_baseline_proposal_comparison.py`
- `tools/audit_baseline_implementations.py`
- `docs/BASELINE_PROPOSAL_COMPARISON_GUIDE.md`
- `docs/BASELINE_IMPLEMENTATION_AUDIT.md`

The comparison runner performs multi-seed training, aggregation, strict
validation, terminal reporting, LaTeX export, and proposal-versus-best-baseline
delta calculation. No runtime source patching or inline Python heredocs are
used by the Kaggle scripts.

## Baseline implementation audit

All eleven retained baselines passed:

- fair-config availability and metadata;
- official-faithful config availability and metadata;
- defining paper-level module checks;
- model construction;
- finite forward pass;
- binary segmentation output shape;
- fair loss compatibility;
- backward-pass and finite-gradient tests;
- paper-style auxiliary/boundary output contracts where applicable.

The implementations are paper-aligned reimplementations. They are not claimed
to be byte-for-byte copies of every authors' original repository. The
`configs/fair` family is used for controlled architecture comparison, while
`configs/official_faithful` validates paper-style output and loss behavior.

## Fairness audit

The 13 fair configurations passed the strict protocol audit:

- same image size and batch size;
- same augmentation;
- same optimizer, learning rate, weight decay, and scheduler;
- same epoch count;
- same BCE-Dice objective;
- same threshold and evaluation protocol;
- auxiliary and boundary losses disabled for all methods;
- APDR changes no Plain Fourier setting and adds only `apdr_*` keys.

## Tests

Validation was executed in two non-overlapping groups:

- baseline, implementation, comparison, and faithful-contract group:
  **106 passed**;
- dataset, proposal, runtime, integration, and pipeline group:
  **92 passed**.

Total: **198 tests passed**.

Additional checks passed:

- Python AST parsing for all `.py` files;
- shell syntax validation for `run.sh` and every script in `scripts/*.sh`;
- baseline/proposal synthetic 13-model × 3-seed aggregation and reporting;
- command-line help for the two-proposal, full-ablation, and
  baseline/proposal runners;
- final repository cleanliness audit.

The actual ETIS GPU benchmark was not run in this environment.
