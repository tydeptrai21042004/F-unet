# DAPR-BAF U-Net

`dapr_baf_unet` is Proposal III in this repository. It combines:

- direct, residual-free global amplitude--phase reconstruction;
- overlapping local Fourier windows with normalized raised-cosine overlap-add;
- local amplitude-only refinement, preserving local phase;
- boundary-only routing from the coarse segmentation probability;
- a small nonzero initial refinement gate.

## Main configuration

```bash
python scripts/train_one.py \
  --model dapr_baf_unet \
  --config configs/fair/dapr_baf_unet.yaml \
  --dataset etis
```

## Component ablation

```bash
python scripts/run_dapr_baf_ablation.py \
  --dataset etis \
  --seeds 42,1,2
```

The ablation isolates local refinement, boundary routing, overlapping windows,
global phase modulation, and global/local spectral channel mixing.
