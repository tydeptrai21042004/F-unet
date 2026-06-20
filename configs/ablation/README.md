# Controlled ablation of the two proposed methods

This directory intentionally contains only two configurations:

1. `plain_fourier_unet.yaml` — **Proposal I: Plain Fourier U-Net**;
2. `apdr_fourier_unet.yaml` — **Proposal II: APDR-Fourier U-Net**.

Proposal II contains the complete Proposal-I path unchanged and adds only the
zero-initialized amplitude–phase disagreement-routed residual adapter. The data,
split, augmentation, optimization, loss, epoch count, threshold, and
deterministic settings are identical. Consequently, the comparison isolates the
incremental contribution of the APDR adapter.

3. `dapr_baf_unet.yaml` — **Proposal III: DAPR-BAF U-Net**, using direct
   global amplitude--phase reconstruction and boundary-guided overlapping
   local amplitude refinement.
