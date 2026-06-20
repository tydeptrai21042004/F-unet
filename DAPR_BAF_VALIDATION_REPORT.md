# DAPR-BAF integration validation

## Added proposal

- Registered model: `dapr_baf_unet`
- Alias names: `dapr`, `dapr_baf`
- Main implementation: `src/models/proposal/dapr_baf_unet.py`
- Permanent ablations: `src/models/proposal/dapr_baf_ablation_variants.py`

The model uses residual-free global amplitude--phase reconstruction followed by
boundary-guided, overlapping, local amplitude-only refinement. Local windows
are processed per image in bounded chunks to avoid constructing the full
batch-wide overlapping spectrum at once.

## Permanent ablation scope

| Model | Removed or changed factor | Trainable parameters |
|---|---|---:|
| `dapr_direct_unet` | No local refinement | 9,633,409 |
| `dapr_baf_uniform_route` | No boundary routing | 9,641,186 |
| `dapr_baf_nonoverlap` | Non-overlapping local windows | 9,641,186 |
| `dapr_baf_no_global_phase` | No global phase modulation | 9,641,186 |
| `dapr_baf_no_global_channel_mix` | No global channel mixing | 9,051,362 |
| `dapr_baf_no_local_channel_mix` | No local channel mixing | 9,640,162 |
| `dapr_baf_unet` | Complete model | 9,641,186 |

The global phase tensor remains allocated in the phase-disabled ablation to
preserve the canonical bottleneck implementation; it is inactive in the
forward path.

## Validation performed

- DAPR-BAF configuration and registration audit passed.
- Repository cleanliness audit passed with zero cache violations.
- 19 DAPR-BAF and end-to-end pipeline tests passed.
- 41 existing APDR and full-component-ablation tests passed.
- 32 all-model build, forward, loss, gradient, trainer, and evaluator tests passed.
- Python source parsing and shell-script syntax checks passed.

The exhaustive repository test command contains expensive baseline-runtime
checks and exceeded the execution window without reporting a failure. The
proposal-specific, compatibility, fairness, and pipeline suites completed.
