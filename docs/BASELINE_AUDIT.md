# Retained manuscript baselines

The active fair benchmark contains exactly eleven published comparison
baselines:

1. U-Net
2. U-Net++
3. Attention U-Net
4. PraNet
5. ACSNet
6. HarDNet-MSEG
7. CFANet
8. Polyp-PVT
9. CaraNet
10. HSNet
11. ResUNet++

Two proposed methods are compared against them:

1. Plain Fourier U-Net
2. APDR-Fourier U-Net

## Implementation status

The baseline classes are paper-aligned reimplementations. They expose and test
the defining modules of the named architectures, including attention gates,
reverse-attention branches, RFB aggregation, local/global context modules,
HarDNet/Res2Net/PVT-compatible backbones, boundary fusion, hybrid semantic
modules, residual blocks, squeeze-and-excitation, and ASPP where applicable.

This repository does **not** claim that every implementation is byte-for-byte
identical to each authors' original repository. Two configuration families are
therefore maintained:

- `configs/fair`: one shared data, optimizer, loss, epoch, threshold, and
  evaluation protocol for controlled comparison;
- `configs/official_faithful`: paper-style auxiliary/boundary output and loss
  contracts for implementation validation.

Run the full implementation audit with:

```bash
python tools/audit_baseline_implementations.py
```

The generated detailed report is `docs/BASELINE_IMPLEMENTATION_AUDIT.md`.
The exact manuscript rows supplied for these methods remain in
`docs/manuscript_baselines.tex`.
