# Cross-system comparison matrix (all metrics, one-shot aggregate)

12 runs total | success 10 | failed 2

> **Speed boundary reminder**: to compare inference speed use "DiT sampling ms" (component-level denoise time, reliable). "end-to-end ms" includes one-time on-the-fly quantization conversion / model loading (see the "boundary" column: net-inference = excludes load/encoding, incl-load+encode = single CLI run), and must not be used for cross-system speed claims. Quantization quality loss (PSNR/SSIM/LPIPS vs FP16) is only meaningful within the same system vs its own FP16 baseline; not comparable across systems.

> **Special note for sd.cpp**: stable-diffusion.cpp loads layer-by-layer while sampling, and on-the-fly quantization conversion (q4_K/q8, tens to hundreds of seconds) folds into the denoise-stage timing, so its "DiT sampling ms" is likewise inflated under quantized tiers and does not represent pure inference. sd.cpp speed should be re-measured with pre-quantized weights, or only used as a same-tier trend reference; it cannot be compared directly with edge/diffusers DiT sampling.

> **The headline tier is q8** (usable image quality); q4 is only an extreme VRAM-saving reference point with obvious quality loss, and is not suitable for speed/quality advantage claims.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | video_01 | success | 32922.0 | 94438.9 | incl-load+encode | 354.1 | 5003.6 | 17776 | 16802 | 17776 | 17750 | 0.307 | 5.99 | 0.025 | 0.902 | 0.011 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(1)** | 32922.0 | 94438.9 |  | 354.1 | 5003.6 | 17776 | 16802 | 17776 | 17750 | 0.307 | 5.99 | 0.025 | 0.902 | 0.011 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | video_01 | success | 36169.0 | 107779.6 | incl-load+encode | 317.6 | 5101.6 | 9366 | 8270 | 9366 | 9340 | 0.297 | 5.74 | 0.019 | 0.941 | 0.012 | 14.69 | 0.502 | 0.313 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 36169.0 | 107779.6 |  | 317.6 | 5101.6 | 9366 | 8270 | 9366 | 9340 | 0.297 | 5.74 | 0.019 | 0.941 | 0.012 | 14.69 | 0.502 | 0.313 |
| edge-dit.cpp | q8_0 | no-offload | none | video_01 | success | 36119.5 | 52362.2 | incl-load+encode | 309.1 | 5000.3 | 12244 | 11148 | 12244 | 12218 | 0.307 | 5.95 | 0.024 | 0.906 | 0.011 | 23.02 | 0.900 | 0.070 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 36119.5 | 52362.2 |  | 309.1 | 5000.3 | 12244 | 11148 | 12244 | 12218 | 0.307 | 5.95 | 0.024 | 0.906 | 0.011 | 23.02 | 0.900 | 0.070 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | video_01 | success | 238552.1 | 478163.6 | incl-load+encode | 1942.4 | 5342.9 | 17584 | 5352 | 17584 | 3688 | 0.229 | 3.84 | 0.269 | 0.564 | 0.075 | — | — | — |
| **edge-dit.cpp** | **f16** | **full offload (max-vram 20g)** | **none** | **mean** | **(1)** | 238552.1 | 478163.6 |  | 1942.4 | 5342.9 | 17584 | 5352 | 17584 | 3688 | 0.229 | 3.84 | 0.269 | 0.564 | 0.075 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | video_01 | failed | — | — | process-level co | — | — | 556 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | video_01 | success | 199778.5 | 230415.0 | incl-load+encode | 1393.1 | 5254.0 | 19958 | 19958 | 18562 | 17320 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | — | — | — |
| **edge-dit.cpp** | **f16->q8_0(auto-fit)** | **te offload + vae offload (max-vram 20g) (auto-fit)** | **none** | **mean** | **(1)** | 199778.5 | 230415.0 |  | 1393.1 | 5254.0 | 19958 | 19958 | 18562 | 17320 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | video_01 | success | 198495.1 | 501570.8 | incl-load+encode | 319.1 | 5023.2 | 18396 | 16146 | 18396 | 17014 | 0.236 | 4.39 | 0.037 | 0.854 | 0.018 | 8.95 | 0.048 | 1.231 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 198495.1 | 501570.8 |  | 319.1 | 5023.2 | 18396 | 16146 | 18396 | 17014 | 0.236 | 4.39 | 0.037 | 0.854 | 0.018 | 8.95 | 0.048 | 1.231 |
| edge-dit.cpp | q8_0 | no-offload | none | video_01 | failed | — | — | process-level co | — | — | 23962 | — | — | — | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | video_01 | success | 199598.3 | 246207.8 | incl-load+encode | 1398.0 | 5270.7 | 20028 | 20028 | 18562 | 17320 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | 7.89 | 0.045 | 1.008 |
| **edge-dit.cpp** | **q8_0** | **te offload + vae offload (max-vram 20g) (auto-allocate)** | **none** | **mean** | **(1)** | 199598.3 | 246207.8 |  | 1398.0 | 5270.7 | 20028 | 20028 | 18562 | 17320 | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | 7.89 | 0.045 | 1.008 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | prompt | status | DiT sampling ms | end-to-end ms | boundary | TE_ms | VAE_ms | peak VRAM | TE VRAM | DiT VRAM | VAE VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | video_01 | success | 6857.2 | 31963.6 | incl-load+encode | 241.1 | 6541.6 | 17760 | 16780 | 17760 | 17734 | 0.325 | 5.89 | 0.076 | 0.842 | 0.009 | — | — | — |
| **edge-dit.cpp** | **f16** | **no-offload** | **none** | **mean** | **(1)** | 6857.2 | 31963.6 |  | 241.1 | 6541.6 | 17760 | 16780 | 17760 | 17734 | 0.325 | 5.89 | 0.076 | 0.842 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | video_01 | success | 7489.3 | 77779.6 | incl-load+encode | 187.6 | 5818.8 | 9350 | — | 9350 | 9324 | 0.331 | 6.11 | 0.050 | 0.873 | 0.007 | 11.35 | 0.356 | 0.454 |
| **edge-dit.cpp** | **q4_k** | **no-offload** | **none** | **mean** | **(1)** | 7489.3 | 77779.6 |  | 187.6 | 5818.8 | 9350 | — | 9350 | 9324 | 0.331 | 6.11 | 0.050 | 0.873 | 0.007 | 11.35 | 0.356 | 0.454 |
| edge-dit.cpp | q8_0 | no-offload | none | video_01 | success | 7483.6 | 23839.7 | incl-load+encode | 184.1 | 4979.7 | 12228 | — | 12228 | 12202 | 0.323 | 5.90 | 0.081 | 0.830 | 0.009 | 18.02 | 0.707 | 0.163 |
| **edge-dit.cpp** | **q8_0** | **no-offload** | **none** | **mean** | **(1)** | 7483.6 | 23839.7 |  | 184.1 | 4979.7 | 12228 | — | 12228 | 12202 | 0.323 | 5.90 | 0.081 | 0.830 | 0.009 | 18.02 | 0.707 | 0.163 |