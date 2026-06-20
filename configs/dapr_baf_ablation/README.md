# DAPR-BAF component ablation

This directory contains the fixed one-factor ablation of Proposal III,
**DAPR-BAF U-Net**:

1. `dapr_direct_unet`: direct global amplitude--phase reconstruction only;
2. `dapr_baf_uniform_route`: removes boundary routing;
3. `dapr_baf_nonoverlap`: replaces overlapping windows with non-overlapping windows;
4. `dapr_baf_no_global_phase`: removes global phase modulation;
5. `dapr_baf_no_global_channel_mix`: removes global spectral channel mixing;
6. `dapr_baf_no_local_channel_mix`: removes local spectral channel mixing;
7. `dapr_baf_unet`: complete model.

All configurations share the same data, optimization, loss, and evaluation
protocol. Only the named architectural factor changes.
