# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.307 | 5.99 | 0.025 | 0.902 | 0.011 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.297 | 5.74 | 0.019 | 0.941 | 0.012 | 14.69 | 0.502 | 0.313 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.307 | 5.95 | 0.024 | 0.906 | 0.011 | 23.02 | 0.900 | 0.070 |

## wan2-t2v-14b-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.229 | 3.84 | 0.269 | 0.564 | 0.075 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | f16->q8_0(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.236 | 4.39 | 0.037 | 0.854 | 0.018 | 8.95 | 0.048 | 1.231 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — | — | — |
| edge-dit.cpp | q8_0 | te offload + vae offload (max-vram 20g) (auto-allocate) | none | 0.240 | 4.98 | 0.016 | 0.908 | 0.009 | 7.89 | 0.045 | 1.008 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.325 | 5.89 | 0.076 | 0.842 | 0.009 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.331 | 6.11 | 0.050 | 0.873 | 0.007 | 11.35 | 0.356 | 0.454 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.323 | 5.90 | 0.081 | 0.830 | 0.009 | 18.02 | 0.707 | 0.163 |
