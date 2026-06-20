# Integrated full-component ablation validation

## Scope

The repository now permanently supports 14 controlled variants:

- one U-Net spatial reference;
- five Proposal-I variants including complete Plain Fourier U-Net;
- eight Proposal-II variants including complete APDR-Fourier U-Net.

No Kaggle-time source patching or inline Python heredoc is required.

## Permanent implementation

- `src/models/proposal/full_ablation_variants.py`
- `configs/full_component_ablation/*.yaml`
- `tools/audit_full_component_ablation.py`
- `scripts/aggregate_training_results.py`
- `scripts/validate_ablation_results.py`
- `scripts/report_full_component_ablation.py`
- `scripts/run_full_component_ablation.py`
- `scripts/kaggle_full_component_ablation_etis_3seeds.sh`
- `KAGGLE_FULL_COMPONENT_ABLATION_CELL.sh`

## Validation performed

- Full fairness and one-factor-change audit: passed.
- Python compilation: passed.
- Shell syntax validation: passed.
- Repository tests: 193 passed, executed in three groups.
- Synthetic 14-model x 3-seed reporting pipeline:
  - 42 test records aggregated;
  - 42 training records aggregated;
  - strict result validation passed;
  - terminal result table generated;
  - LaTeX table generated;
  - contribution-delta CSV generated.

The actual 42 GPU training/evaluation runs were not executed in this environment.
