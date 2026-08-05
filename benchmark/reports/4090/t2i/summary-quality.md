# Quality summary (mean)

> Quality columns are per task (t2i: CLIP/aesthetic/IR; editing: dir-CLIP/keep-SSIM/keep-LPIPS/aesthetic/IR; video: per-frame CLIP/aesthetic + temporal). PSNR↑/SSIM↑/LPIPS↓ are quantization vs the same system's own FP16 baseline (not comparable across systems). Baseline tiers show —.


## flux-dev-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | full offload | none | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.308 | 6.14 | 1.737 | — | — | — |
| diffusers | w8 | no-offload | none | 0.307 | 6.12 | 1.735 | 30.81 | 0.970 | 0.024 |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.302 | 6.11 | 1.582 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.304 | 6.03 | 1.611 | 21.45 | 0.808 | 0.243 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.296 | 6.07 | 1.630 | 28.65 | 0.892 | 0.130 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.309 | 6.06 | 1.476 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.311 | 6.05 | 1.525 | 23.52 | 0.878 | 0.143 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.310 | 6.10 | 1.525 | 29.00 | 0.962 | 0.043 |

## flux-schnell-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | w8 | no-offload | none | 0.306 | 5.86 | 1.597 | — | — | — |
| edge-dit.cpp | f16 | full offload (max-vram 20g) | none | 0.331 | 5.91 | 1.552 | — | — | — |
| edge-dit.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.327 | 6.02 | 1.113 | 17.64 | 0.697 | 0.262 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.328 | 5.90 | 1.414 | 24.84 | 0.879 | 0.081 |
| stable-diffusion.cpp | f16 | full offload (max-vram 20g) | none | 0.320 | 5.88 | 1.735 | — | — | — |
| stable-diffusion.cpp | f16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.318 | 5.89 | 1.708 | 16.67 | 0.668 | 0.347 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.324 | 5.86 | 1.732 | 22.45 | 0.867 | 0.105 |

## qwen-image-lightning-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 0.333 | 6.05 | 1.837 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.336 | 5.86 | 1.832 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.339 | 5.96 | 1.844 | 20.43 | 0.789 | 0.175 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## qwen-image-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | — | — | — | — | — | — |
| diffusers | bf16 | sequential (full offload) | none | 0.326 | 5.98 | 1.831 | — | — | — |
| diffusers | w8 | full offload | none | 0.322 | 6.02 | 1.834 | 25.95 | 0.906 | 0.078 |
| diffusers | w8 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | bf16 | full offload (max-vram 20g) | none | 0.327 | 5.92 | 1.848 | — | — | — |
| edge-dit.cpp | bf16 | no-offload | none | — | — | — | — | — | — |
| edge-dit.cpp | f16->q4_k(auto-fit) | te offload + vae offload (max-vram 20g) (auto-fit) | none | 0.334 | 5.84 | 1.864 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.327 | 5.96 | 1.850 | 24.12 | 0.864 | 0.120 |
| edge-dit.cpp | q8_0 | DiT offload + te offload (max-vram 20g) (auto-allocate) | none | 0.325 | 5.96 | 1.855 | 28.79 | 0.912 | 0.073 |
| edge-dit.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | bf16 | full offload (max-vram 20g) | none | 0.328 | 6.07 | 1.840 | — | — | — |
| stable-diffusion.cpp | bf16 | no-offload | none | — | — | — | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.320 | 5.93 | 1.847 | 19.87 | 0.770 | 0.200 |
| stable-diffusion.cpp | q8_0 | full offload (max-vram 20g) | none | 0.325 | 6.02 | 1.833 | 28.33 | 0.946 | 0.029 |
| stable-diffusion.cpp | q8_0 | no-offload | none | — | — | — | — | — | — |

## sd3-medium-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.336 | 5.52 | 1.685 | — | — | — |
| diffusers | w8 | no-offload | none | 0.336 | 5.68 | 1.660 | 22.08 | 0.869 | 0.140 |
| edge-dit.cpp | f16 | no-offload | none | 0.333 | 5.72 | 1.652 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.345 | 5.30 | 1.388 | 17.87 | 0.704 | 0.406 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.328 | 5.70 | 1.556 | 22.19 | 0.873 | 0.144 |
| stable-diffusion.cpp | f16 | no-offload | none | 0.329 | 5.66 | 1.654 | — | — | — |
| stable-diffusion.cpp | q4_k | no-offload | none | 0.336 | 5.57 | 1.448 | 15.61 | 0.670 | 0.451 |
| stable-diffusion.cpp | q8_0 | no-offload | none | 0.341 | 5.58 | 1.356 | 23.30 | 0.853 | 0.180 |

## sd35-medium-turbo-text-to-image  (text-to-image)

| system | precision | budget | cache | CLIP | aesthetic | IR | PSNRvsFP16 | SSIMvsFP16 | LPIPSvsFP16 |
|---|---|---|---|---|---|---|---|---|---|
| diffusers | bf16 | no-offload | none | 0.322 | 5.37 | -0.017 | — | — | — |
| diffusers | w8 | no-offload | none | 0.322 | 5.26 | 0.425 | 35.20 | 0.972 | 0.037 |
| edge-dit.cpp | f16 | no-offload | none | 0.337 | 5.45 | 1.612 | — | — | — |
| edge-dit.cpp | q4_k | no-offload | none | 0.326 | 5.51 | 1.049 | 19.38 | 0.771 | 0.292 |
| edge-dit.cpp | q8_0 | no-offload | none | 0.339 | 5.47 | 1.196 | 28.03 | 0.908 | 0.115 |
