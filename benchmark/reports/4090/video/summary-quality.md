# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.317 | 5.36 | 0.072 | 0.873 | 0.010 | — | — | — |
| diffusers | w8 | no-offload | none | 0.320 | 5.45 | 0.066 | 0.872 | 0.011 | 19.76 | 0.766 | 0.264 |
| edge-dit.cpp | f16 | no-offload | none | 0.288 | 5.51 | 0.016 | 0.938 | 0.008 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.292 | 5.67 | 0.023 | 0.911 | 0.012 | 18.41 | 0.674 | 0.286 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.320 | 5.38 | 0.012 | 0.941 | 0.004 | 26.43 | 0.853 | 0.104 |
| stable-diffusion.cpp | f16 | no-offload | none | 0.315 | 5.59 | 0.032 | 0.893 | 0.014 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.308 | 5.64 | 0.032 | 0.875 | 0.017 | 18.91 | 0.650 | 0.267 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.330 | 5.61 | 0.024 | 0.901 | 0.006 | 25.34 | 0.821 | 0.120 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 0.310 | 5.58 | 0.064 | 0.841 | 0.020 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.316 | 5.68 | 0.056 | 0.867 | 0.023 | 12.77 | 0.428 | 0.405 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.311 | 5.56 | 0.059 | 0.860 | 0.021 | 17.21 | 0.635 | 0.240 |
