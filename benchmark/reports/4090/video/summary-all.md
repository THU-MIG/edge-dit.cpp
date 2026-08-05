# Summary table (mean, core columns)

> One table at a glance, split by task (quality columns differ per task). For speed look at DiT sampling ms; VRAM unit MiB; PSNR/SSIM/LPIPS are quantization vs same-system FP16.


## wan2-t2v-1.3b-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 53707.8 | 56728.5 | 20498 | 0.317 | 5.36 | 0.072 | 0.873 | 0.010 | — | — | — |
| diffusers | w8 | no-offload | none | 56580.2 | 59598.0 | 19568 | 0.320 | 5.45 | 0.066 | 0.872 | 0.011 | 19.76 | 0.766 | 0.264 |
| edge-dit.cpp | f16 | no-offload | none | 49444.8 | 55423.7 | 17708 | 0.288 | 5.51 | 0.016 | 0.938 | 0.008 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 54042.6 | 60005.0 | 9298 | 0.292 | 5.67 | 0.023 | 0.911 | 0.012 | 18.41 | 0.674 | 0.286 |
| edge-dit.cpp | q8_0 | no-offload | none | 53963.7 | 59927.0 | 12176 | 0.320 | 5.38 | 0.012 | 0.941 | 0.004 | 26.43 | 0.853 | 0.104 |
| stable-diffusion.cpp | f16 | no-offload | none | 83423.3 | 105119.1 | 17818 | 0.315 | 5.59 | 0.032 | 0.893 | 0.014 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 86783.3 | 125168.3 | 11372 | 0.308 | 5.64 | 0.032 | 0.875 | 0.017 | 18.91 | 0.650 | 0.267 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 80383.3 | 111749.9 | 11308 | 0.330 | 5.61 | 0.024 | 0.901 | 0.006 | 25.34 | 0.821 | 0.120 |

## wan21-t2v-1.3b-distill-text-to-video  (text-to-video)

| system | precision | budget | cache | DiTms | end-to-end ms | peak VRAM | frame CLIP | frame aesthetic | temporal LPIPS | temporal SSIM | flicker std | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | f16 | no-offload | none | 6732.4 | 12594.0 | 17692 | 0.310 | 5.58 | 0.064 | 0.841 | 0.020 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 7394.2 | 13198.0 | 9282 | 0.316 | 5.68 | 0.056 | 0.867 | 0.023 | 12.77 | 0.428 | 0.405 |
| edge-dit.cpp | q8_0 | no-offload | none | 7384.4 | 13132.7 | 12160 | 0.311 | 5.56 | 0.059 | 0.860 | 0.021 | 17.21 | 0.635 | 0.240 |
