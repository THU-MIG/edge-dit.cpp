# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 32922.0 | 94438.9 | 17776 | 0.307 | 5.99 | 0.025 | 0.902 | 0.011 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 36169.0 | 107779.6 | 9366 | 0.297 | 5.74 | 0.019 | 0.941 | 0.012 | 14.69 | 0.502 | 0.313 |
| edge-dit.cpp | q8_0 | no-offload | none | 36119.5 | 52362.2 | 12244 | 0.307 | 5.95 | 0.024 | 0.906 | 0.011 | 23.02 | 0.900 | 0.070 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 238552.1 | 478163.6 | 17584 | 0.229 | 3.84 | 0.269 | 0.564 | 0.075 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | 556 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 199778.5 | 230415.0 | 19958 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 198495.1 | 501570.8 | 18396 | 0.236 | 4.39 | 0.037 | 0.854 | 0.018 | 8.95 | 0.048 | 1.231 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | 23962 | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 199598.3 | 246207.8 | 20028 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | 7.89 | 0.045 | 1.008 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 6857.2 | 31963.6 | 17760 | 0.325 | 5.89 | 0.076 | 0.842 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 7489.3 | 77779.6 | 9350 | 0.331 | 6.11 | 0.050 | 0.873 | 0.007 | 11.35 | 0.356 | 0.454 |
| edge-dit.cpp | q8_0 | no-offload | none | 7483.6 | 23839.7 | 12228 | 0.323 | 5.90 | 0.081 | 0.830 | 0.009 | 18.02 | 0.707 | 0.163 |
