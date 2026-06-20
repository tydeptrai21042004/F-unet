# Retained manuscript baselines

The active benchmark contains only the following eleven published baselines:

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

The fair configurations use a single shared data, optimization and evaluation
protocol. Architecture-specific auxiliary and boundary outputs remain available
inside their implementations, but they are not included in the controlled fair
loss. The exact manuscript rows supplied for these methods are stored in
`docs/manuscript_baselines.tex`.
