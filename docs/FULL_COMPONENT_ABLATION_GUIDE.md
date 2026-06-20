# Full component ablation

The repository permanently supports a controlled 14-model study of the two proposed methods. No Kaggle-time source patching is needed.

## Proposal I variants

- `plain_fourier_amplitude_only`
- `plain_fourier_phase_only`
- `plain_fourier_no_channel_mix`
- `plain_fourier_no_residual`
- `plain_fourier_unet`

## Proposal II variants

- `apdr_uniform_route`
- `apdr_no_disagreement`
- `apdr_no_uncertainty`
- `apdr_no_boundary`
- `apdr_no_context`
- `apdr_local_amplitude_only`
- `apdr_local_phase_only`
- `apdr_fourier_unet`

`unet` is included as the spatial-backbone reference. All 14 configurations share exactly the same data, training, and evaluation protocol.

## Local audit

```bash
python tools/audit_full_component_ablation.py
python -m pytest -q tests/test_full_component_ablation.py
```

## Full ETIS experiment

```bash
bash scripts/kaggle_full_component_ablation_etis_3seeds.sh
```

The end-to-end runner invokes permanent Python scripts for training aggregation, strict output validation, terminal reporting, contribution-delta export, and LaTeX-table generation.
